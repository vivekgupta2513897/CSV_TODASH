import openai
import uuid
import json
import autogen
import matplotlib as plt
import subprocess
import sys
import openpyxl # type: ignore
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from openai import OpenAI
import os

chart_filename = f"{str(uuid.uuid4())}_chart.png"
code_language= 'python'
chart_location = os.path.join('plots', 'Charts')
dashboard_location = os.path.join('plots', 'Dashboard')
os.makedirs(chart_location, exist_ok=True)
os.makedirs(dashboard_location, exist_ok=True)

config_list = autogen.config_list_from_json("apikey.json")

query_processing_agent = autogen.AssistantAgent(
    name="Query_Processor",
    llm_config={"config_list": config_list},
    system_message=f"""**Role**: You are a query processing agent who interprets and processes user queries. You have the following tasks:
- Parse the user's query and identify key data points or requirements.
- Translate the query into a structured format (e.g., SQL, filters) to extract relevant data.
- Ensure the query applies the correct filters (e.g., date range, category) and targets the right data.
- Provide the user with the results in a clear and structured format.
- Break down complex queries into smaller parts, if necessary, to ensure each part is correctly executed and aggregated.
- If the query involves aggregations, ensure the proper functions (e.g., SUM, COUNT, AVG) are used and explain the results clearly.
- Suggest improvements or optimizations to the query if necessary, such as indexing or performance improvements.
- If the query involves time-sensitive or real-time data, ensure that the appropriate filters (e.g., timestamps) are used to return accurate results.
- After processing, confirm the completion of the task and ask if the user needs further assistance or additional queries.
"""
)

planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    system_message=f"""You are a Planner Agent responsible for creating all types of charts based on user requests.
    - Load the data from the uploaded file.
    - Determine the type of analysis (basic, advanced, forecasting, or AI/ML).
    - Clean the data by handling missing values, duplicates, or formatting issues.
    - Choose the best chart type based on the analysis.
    - Customize the chart (e.g., axis labels, title).
    - Add insights or annotations to highlight trends if needed.
    Save the plot at the specified location {chart_location} and print the filename.
    - Ensure that data security and privacy are maintained throughout the process.
    - **DO NOT GENERATE CODE YOURSELF.** Instruct the CodeWriter to generate the necessary code.
            """
        )

data_retrival_agent = autogen.AssistantAgent(
    name="Data_Retriever",
    llm_config={"config_list": config_list},
    system_message=f"""
    **Role**:You are a data retrival agent who retrives data from the file that has been uploaded.You have following tasks:
- Retrieve the most recent data from the file. 
- Focus on the following data points: specific variables. 
- Ensure that the data is cleaned by handling missing values, duplicates, or outliers, and is formatted in a desired format (e.g., CSV, JSON, or other suitable formats for analysis).
- Provide a summary of the data retrieved, including a brief description of the dataset, the number of rows/columns, and any noteworthy characteristics (e.g., missing data, duplicates, etc.).
- Ensure the data is structured and ready for analysis, including appropriate column names, and consistent formatting.
- Analyze and document the distribution of numerical variables to detect any unusual or extreme values that might need attention.
- Identify and provide a breakdown of missing data percentages and handle missing values (e.g., imputation, removal, or flagging).
- Ensure that categorical variables are properly encoded (e.g., one-hot encoding or label encoding if required).
- Provide an overview of the data, highlighting any trends or patterns observed in the initial analysis (e.g., time trends, correlations, etc.).
"""
)

code_writer = autogen.AssistantAgent(
    name="code_writer",
    llm_config={"config_list": config_list},
    system_message=f"""Write Python code to generate charts dynamically based on user queries.  
- Ensure the plot is saved in a temporary directory and can be accessed by the dashboard.  
- Print "Chart generated successfully" after creating the plot.  
- Handle missing or incorrect user inputs gracefully by providing clear error messages and suggestions.  
- Adapt the approach based on the analysis type (basic, forecasting, AI/ML).  
- Choose the most suitable visualization type based on the data and user query.  
- Allow customization options like color schemes, labels, titles, and annotations.  
- Ensure interactive elements (e.g., tooltips, zooming) when using libraries like Plotly or Altair.  
- Optimize the code for efficiency, avoiding unnecessary computations or redundant processing.  
- Handle large datasets efficiently by sampling or aggregating when necessary.  
- Log the chart creation process for debugging and traceability.  
- Verify that the generated chart is clear, readable, and provides meaningful insights.  
- Suggest alternative chart types if the requested visualization isn’t ideal for the given data.  
- Ensure compatibility with different file formats (CSV, Excel, JSON).  
- Implement best practices for accessibility, making charts readable for all users.
-  Ensure to save the plot using plt.save and store it at the specified {chart_location}. Print "Chart saved as {chart_filename}" after saving the plot.  

            """
        )

code_executor = UserProxyAgent(
            name="code_executor",
            human_input_mode="NEVER",
            code_execution_config={
        "work_dir": "plotter_code",
        "use_docker": False  
            },
            llm_config={"config_list": config_list},
            system_message=f'''You execute the code provided by CodeWriter and return results.
- Always validate the code before execution.
- Display results in a structured format (tabular for data, graphical for charts).
- Ensure plots are saved in a **temporary directory** and accessible to the dashboard.
- Provide meaningful error messages for debugging.
- Handle unexpected errors gracefully and suggest possible solutions.
- Ensure the generated code is optimized for performance and avoids redundant computations.
- Log execution steps and errors for better traceability and debugging.
- Verify that dependencies and required libraries are correctly loaded before execution.
- Ensure security by preventing execution of unsafe or malicious code.
- Confirm the output matches user expectations and suggest refinements if necessary.
- In cases where a graphical output is produced, ensure the plot is saved using plt.save and stored at the specified {chart_location}, printing "Chart saved as {chart_filename}" after saving.
            '''
        )

debugger = autogen.AssistantAgent(
    name="debugger",
    llm_config={"config_list": config_list},
    system_message=f"""
   You are responsible for identifying and fixing errors in the generated code.
    - If errors occur, analyze the traceback and apply fixes.
    - Re-run the corrected code and verify output.
    - Ensure all required libraries are correctly imported.
    - test the code after execution to ensure it works as expected and correct any issues that arise.
    - If the code fails or produces unexpected results, debug it step-by-step, fix the problem.
    - Re-run the code until it's fully functional.Confirm that the plot is correctly generated and saved, and the output is accurate when done resolving errors.
    - Ensure that all dependencies and libraries required for execution are correctly installed and imported.
    - Ensure that the plot is saved using plt.save and stored at the specified {chart_location}, printing "Chart saved as {chart_filename}" once the plot is saved
"""
        )

Error_Handling_Agent = autogen.AssistantAgent(
    name="Error_Handler",
    llm_config={"config_list": config_list},
    system_message=f"""You are an agent handling errors for missing or incorrect user inputs. 
    - When a file is missing or invalid, instruct the user to upload a valid file (e.g., CSV, JSON, Excel). Example: "Please upload a valid file (CSV, JSON, Excel, etc.)."
    - If the user fails to provide a query or required field, prompt them to fill it in. Example: "Please enter a query for the visualization."
    - For unsupported file formats, tell the user to upload the correct type. Example: "Please upload a CSV, Excel, or JSON file."
    - When data is missing or formatted incorrectly, point out the issue. Example: "The 'Date' column is missing. Please check your file."
    - For invalid date or time formats, provide the correct format. Example: "Please use the format YYYY-MM-DD for dates."
    - If a required field is missing, ask the user to provide it. Example: "Please specify a chart type."
    - For incorrect numerical input, ask for a valid number. Example: "Please enter a valid number."
    - When a value is out of range, inform the user and provide the acceptable range. Example: "Please provide a number between 1 and 100."
    - If the query is unclear, ask the user to rephrase it. Example: "Please clarify your query (e.g., 'Show sales by region')."
    - Always provide clear instructions for correcting the input and encourage the user to try again.
    """
    )

dashboarding_agent = autogen.AssistantAgent(
    name="Dashboard",
    llm_config={"config_list": config_list},
    system_message=f"""
    You are an agent responsible for creating interactive dashboards. Your task is to generate a user-friendly, visually appealing dashboard that integrates multiple data visualizations. The dashboard should:

1. Include all relevant visualizations (charts, graphs) generated from the provided data.
2. Be organized in a clear and logical layout with sections/tabs for different views.
3. Provide filters and interactive elements where appropriate (e.g., date range picker, dropdown menus, sliders).
4. Display the visualizations in an organized manner, such as in columns, rows, or grids.
5. Ensure that all charts and graphs are responsive and easy to understand.
6. Include explanations, titles, or tooltips to provide context for each visualization.
7. Have consistent styling (color schemes, fonts, etc.) across the dashboard for a polished look.
8. Allow the user to interact with the visualizations (e.g., hover effects, zoom, pan).
9. The dashboars that are created should be saved at {dashboard_location}

"""
        )

UI_agent = autogen.AssistantAgent(
    name="User_interface",
    llm_config={"config_list": config_list},
    system_message=f"""
    You are an agent responsible for designing and developing a user-friendly, visually appealing **User Interface (UI)** for an interactive web application.
    Your task is to create a well-structured, responsive, and intuitive UI that enhances user experience. The UI should:
- Use clear layouts, sections, or tabs for better navigation.  
- Implement buttons, dropdown menus, sliders, input fields, and other interactive elements.  
- Ensure the interface is mobile-friendly and adapts to different screen sizes.  
- Apply a uniform color scheme, typography, and styling for consistency.  
- Use tooltips, labels, and modals to provide contextual information.  
- Incorporate hover effects, transitions, and animations for a dynamic experience.  
- Ensure smooth interactions and fast load times for optimal performance.  
- Display charts, graphs, and other visual elements in a structured manner if applicable.  
- Enable personalization options such as themes, adjustable settings, or saved preferences.  
- Structure, modularize, and make the code ready for deployment using an appropriate framework.  
""")

process_completion = autogen.AssistantAgent(
    name="process_completion",
    llm_config={"config_list": config_list},
    system_message=f"""You are a Process completion agent which takes care that all processes are being completed by all other agents.
    Respond back with information in a tabular format or sequential steps, depending on the context.
    Always provide tabular responses in Markdown format.
    display the data head in Markdown format.
    Ensure all tabular data is processed and presented using Markdown for clarity.
    Give complete details at each step, ensuring clarity in every action for sequential data.
    Transform the data into a more usable format(If required) (e.g., scaling or encoding categorical variables).
    Provide an overview of any transformations applied, explaining the changes made to the data.
    If the process is incomplete, ask the user if they would like to continue or need further assistance.
    Once everything is finished, confirm completion with the user.
    Recommend two new questions or tasks to keep the conversation engaging and move forward.
""")

from autogen.agentchat.agent import Agent

# State Transition Function
def state_transition(last_speaker : Agent, groupchat : GroupChat):
    messages = groupchat.messages
    if len(messages) <= 1:
        return query_processing_agent  
    if last_speaker is query_processing_agent:
        return planner  
    elif last_speaker is planner:
        return data_retrival_agent 
    elif last_speaker is data_retrival_agent:
        return code_writer  
    elif last_speaker is code_writer:
        return code_executor  
    elif last_speaker is code_executor:
        if "exitcode: 1" in messages[-1]["content"]:
            return debugger  
        else:
            return dashboarding_agent 
    elif last_speaker is debugger:
        return Error_Handling_Agent  
    elif last_speaker is Error_Handling_Agent:
        return code_writer 
    elif last_speaker is dashboarding_agent:
        return UI_agent  
    elif last_speaker is UI_agent:
        return process_completion  
    return None  

cs_groupchat = GroupChat(
    agents=[
        query_processing_agent,
        planner,  
        data_retrival_agent,
        dashboarding_agent,
        UI_agent,
        Error_Handling_Agent,
        code_writer,  
        code_executor,  
        debugger,  
        process_completion 
    ],
    speaker_selection_method=state_transition,
    messages=[],
    max_round=900)

cs_manager = GroupChatManager(cs_groupchat,llm_config=config_list[0],)

created_charts = []

def update_created_charts_list():
    """Scan the Charts directory and update the created_charts list with any found charts."""
    global created_charts
    charts_dir = "Charts"
    if os.path.exists(charts_dir) and os.path.isdir(charts_dir):
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith(".png")]
        created_charts = [{"filename": f, "title": f.replace(".png", "").replace("_", " ").title()} for f in chart_files]


def generate_dashboard(visualizations, title="Generated Dashboard", description=""):
    """
    Creates a comprehensive dashboard with all requested visualizations and saves them as PNG files.
    
    Parameters:
    visualizations (list): List of dictionaries with format:
                          {'type': chart_type, 'fig': matplotlib_figure, 'title': chart_title}
    title (str): Dashboard title
    description (str): Dashboard description
    
    Returns:
    list: Paths to the saved PNG files
    """

    # Ensure directories exist
    chart_location = os.path.join('plots', 'Charts')
    dashboard_location = os.path.join('plots', 'Dashboard')
    os.makedirs(chart_location, exist_ok=True)
    os.makedirs(dashboard_location, exist_ok=True)

    # Generate timestamp and ID for file naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_id = str(uuid.uuid4())[:8]

    saved_files = []  # List to store paths of saved PNG files

    # Save each visualization as a separate PNG
    for i, viz in enumerate(visualizations):
        if isinstance(viz, dict) and 'fig' in viz and viz['fig'] is not None:
            filename = f"{timestamp}_{dashboard_id}_chart_{i+1}.png"
            file_path = os.path.join(dashboard_location, filename)

            viz['fig'].tight_layout()  # ✅ Prevent figure cutoff
            viz['fig'].savefig(file_path, format='png', bbox_inches='tight')  # ✅ Save as PNG
            plt.close(viz['fig'])  # ✅ Close figure after saving
            saved_files.append(file_path)  # ✅ Add to saved files list

    # Save summary page as PNG
    summary_filename = f"{timestamp}_{dashboard_id}_summary.png"
    summary_path = os.path.join(dashboard_location, summary_filename)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    ax.text(0.5, 0.9, "Dashboard Summary", fontsize=20, ha='center')

    summary_text = "\n".join(
        [f"- {viz['title']}" for viz in visualizations if isinstance(viz, dict) and 'title' in viz]
    )
    ax.text(0.5, 0.7, summary_text, fontsize=14, ha='center', va='top')

    fig.savefig(summary_path, format='png', bbox_inches='tight')  # ✅ Save as PNG
    plt.close(fig)  # ✅ Close summary figure
    saved_files.append(summary_path)  # ✅ Add to saved files list

    print(f"Dashboard saved as PNG files in: {dashboard_location}")
    return saved_files
def process_user_request(df):
    """Process a user request by generating a dashboard from a given dataframe."""
    global created_charts
    created_charts = []
    generate_dashboard(df)
    st.success("Dashboard generation complete!")



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import os
import datetime
import uuid

# Page configuration
st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_command' not in st.session_state:
    st.session_state.last_command = None
if 'last_chart_type' not in st.session_state:
    st.session_state.last_chart_type = None
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False
if 'generated_charts' not in st.session_state:
    st.session_state.generated_charts = {}

def matching_column(df, term, prefer_categorical=False):
    # First check for exact match
    if term in df.columns:
        return term
    
    # Check for case-insensitive match
    for col in df.columns:
        if col.lower() == term.lower():
            return col
    
    # Check for partial match
    matches = []
    for col in df.columns:
        # Check if term is in column name or column name is in term
        if term.lower() in col.lower() or col.lower() in term.lower():
            similarity = len(set(term.lower()) & set(col.lower())) / len(set(term.lower()) | set(col.lower()))
            matches.append((col, similarity))
    
    # Sort by similarity score
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # If we prefer categorical columns, prioritize them
        if prefer_categorical:
            cat_matches = [m for m in matches if df[m[0]].dtype in ['object', 'category']]
            if cat_matches:
                return cat_matches[0][0]
        
        return matches[0][0]
    
    # If no match, return appropriate column based on preference
    if prefer_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        return cat_cols[0] if not cat_cols.empty else None
    else:
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        return num_cols[0] if not num_cols.empty else None



# Function to extract insights from data
def data_insights(df):
    insights = []
    
    # Basic stats
    insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Check for missing values
    missing_vals = df.isnull().sum().sum()
    if missing_vals > 0:
        missing_pct = (missing_vals / (df.shape[0] * df.shape[1])) * 100
        insights.append(f"Missing values: {missing_vals} ({missing_pct:.2f}%)")
        
        # Columns with highest missing values
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        if cols_with_missing:
            missing_counts = df[cols_with_missing].isnull().sum()
            worst_col = missing_counts.idxmax()
            worst_pct = (missing_counts[worst_col] / df.shape[0]) * 100
            insights.append(f"Column with most missing values: '{worst_col}' ({worst_pct:.2f}%)")
    
    # Numeric column insights
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 0:
        # Find columns with highest and lowest variance
        if len(num_cols) >= 2:
            variances = df[num_cols].var()
            highest_var_col = variances.idxmax()
            lowest_var_col = variances.idxmin()
            insights.append(f"Highest variance: '{highest_var_col}' ({variances[highest_var_col]:.2f})")
            insights.append(f"Lowest variance: '{lowest_var_col}' ({variances[lowest_var_col]:.2f})")
        
        # Find columns with extreme skewness
        skewness = df[num_cols].skew()
        for col, skew_val in skewness.items():
            if abs(skew_val) > 1.0:  # Significant skewness
                direction = "right" if skew_val > 0 else "left"
                insights.append(f"'{col}' is significantly skewed to the {direction} ({skew_val:.2f})")
        
        # Check for extreme outliers using IQR method
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / df.shape[0]) * 100
                if outlier_pct > 5:  # Only report if significant
                    insights.append(f"'{col}' has {len(outliers)} outliers ({outlier_pct:.2f}%)")
    
    # Categorical column insights
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) > 0:
        for col in cat_cols:
            unique_vals = df[col].nunique()
            total_vals = df[col].count()
            
            # Check if cardinality is high
            if unique_vals > 10 and unique_vals / total_vals > 0.5:
                insights.append(f"'{col}' has high cardinality: {unique_vals} unique values ({(unique_vals/total_vals)*100:.2f}%)")
            
            # Check for imbalanced categories
            value_counts = df[col].value_counts()
            if not value_counts.empty:
                most_common = value_counts.index[0]
                most_common_pct = (value_counts.iloc[0] / total_vals) * 100
                
                if most_common_pct > 75:
                    insights.append(f"'{col}' is imbalanced: '{most_common}' represents {most_common_pct:.2f}% of data")
    
    # Check for correlations between numeric columns
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        
        # Get upper triangle mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Apply mask and find high correlations
        high_corrs = corr_matrix.where(mask).stack().sort_values(ascending=False)
        high_corrs = high_corrs[high_corrs < 1.0]  # Remove self-correlations
        
        # Report high correlations
        if not high_corrs.empty and high_corrs.iloc[0] > 0.7:
            col1, col2 = high_corrs.index[0]
            insights.append(f"Strong correlation between '{col1}' and '{col2}' ({high_corrs.iloc[0]:.2f})")
    
    return insights


def create_dashboard(df):
    st.header("📊 Data Insights")
    
    # Generate insights
    insights = data_insights(df)
    
    # Display insights
    for insight in insights:
        st.markdown(f"- {insight}")
    
    st.subheader("📊 Top Categories")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        cat_col = cat_cols[0]  # Use first categorical column
        fig, ax = plt.subplots(figsize=(10, 6))
        df[cat_col].value_counts().head(8).plot.bar(ax=ax)
        plt.title(f'Distribution of {cat_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.subheader("🔄 PCA Visualization")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) >= 3:
        try:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[num_cols])
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_data)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title('PCA - 2 Components')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            st.pyplot(fig)
        except:
            st.warning("Could not create PCA visualization")

# Header
st.title("📊 Agentic Data Analysis Dashboard")

# Main content layout
col1, col2 = st.columns([1, 1])

with col1:
    # File upload section
    st.subheader("1. Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.success(f"✅ File loaded: {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Data preview if data is loaded
    if st.session_state.df is not None:
        st.subheader("2. Data Summary")
        df = st.session_state.df
        
        # Basic info
        st.write(f"📋 Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Data preview
        st.dataframe(df.head(), use_container_width=True)
        
        # Column information
        st.markdown("#### Column Types")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)

with col2:
    # Analysis section
    st.subheader("3. Visualization Command")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Auto Dashboard button
        
        # Command input box
        st.markdown("Enter a command to generate a specific visualization:")
        command = st.text_input("", placeholder="Example: heatmap, bar gender income, scatter loan amount vs income")
        
        st.markdown("""
        **Available commands:**
        - `heatmap` or `correlation`: Creates a correlation heatmap of numeric columns
        - `distribution` or `histogram [column]`: Shows distribution of a column
        - `bar [category] [value]`: Creates a bar chart grouped by category
        - `scatter [x] [y]`: Creates a scatter plot with x and y columns
        - `pie [column]`: Creates a pie chart of a categorical column
        """)
        
        if st.button("Generate Visualization"):
            if not command:
                st.warning("Please enter a visualization command")
            else:
                # Store the command to track changes
                current_command = command.lower().strip()
                
                # Check if this is a new command
                is_new_command = (current_command != st.session_state.last_command)
                
                # Update the last command
                st.session_state.last_command = current_command
                
                # Debug information
                st.info(f"Command: '{current_command}'")
                
                try:
                    # Parse the command to determine chart type
                    chart_type = None
                    if any(word in current_command for word in ["heatmap", "correlation"]):
                        chart_type = "heatmap"
                    elif any(word in current_command for word in ["distribution", "histogram"]):
                        chart_type = "histogram"
                    elif "bar" in current_command:
                        chart_type = "bar"
                    elif "scatter" in current_command:
                        chart_type = "scatter"
                    elif "pie" in current_command:
                        chart_type = "pie"
                    
                    # Debug the chart type
                    st.info(f"Chart type: {chart_type}")
                    
                    # Check if chart type changed
                    is_new_chart = (chart_type != st.session_state.last_chart_type)
                    st.session_state.last_chart_type = chart_type
                    
                    # Clear previous chart if command changed
                    if is_new_command:
                        st.info("New command detected, refreshing visualization")
                    
                    # Extract terms from command
                    command_words = current_command.split()
                    terms = [word for word in command_words if word not in ['bar', 'scatter', 'pie', 'histogram', 'distribution', 'heatmap', 'correlation', 'by', 'vs', 'versus', 'and', 'with', 'the', 'of', 'in', 'on']]
                    
                    # Correlation heatmap
                    if chart_type == "heatmap":
                        numeric_df = df.select_dtypes(include=['int64', 'float64'])
                        if numeric_df.shape[1] > 1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            corr = numeric_df.corr()
                            mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
                            sns.heatmap(corr, annot=True, mask=mask, cmap='coolwarm', ax=ax, fmt=".2f", linewidths=0.5)
                            plt.title('Correlation Heatmap')
                            st.pyplot(fig)
                            st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                        else:
                            st.warning("Not enough numeric columns for correlation analysis")
                    
                    # Distribution/Histogram
                    elif chart_type == "histogram":
                        # Find column name from terms
                        col_name = None
                        if terms:
                            col_name = matching_column(df, terms[0], prefer_categorical=False)
                        else:
                            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            if not num_cols.empty:
                                col_name = num_cols[0]
                            
                        st.info(f"Selected column for histogram: {col_name}")
                        
                        if col_name and col_name in df.columns:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            if pd.api.types.is_numeric_dtype(df[col_name]):
                                sns.histplot(df[col_name].dropna(), kde=True, ax=ax)
                                plt.title(f'Distribution of {col_name}')
                                plt.xlabel(col_name)
                                plt.ylabel('Count')
                            else:
                                # For categorical columns, use countplot
                                sns.countplot(y=df[col_name], ax=ax, order=df[col_name].value_counts().index)
                                plt.title(f'Distribution of {col_name}')
                                plt.xlabel('Count')
                                plt.ylabel(col_name)
                            st.pyplot(fig)
                            st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                        else:
                            st.warning("No suitable column found for distribution analysis")
                    
                    # Bar chart
                    elif chart_type == "bar":
                        cat_col = None
                        num_col = None
                        
                        # If we have at least two terms
                        if len(terms) >= 2:
                            # First term is likely categorical
                            cat_col = matching_column(df, terms[0], prefer_categorical=True)
                            # Second term is likely numerical
                            num_col = matching_column(df, terms[1], prefer_categorical=False)
                            
                            # Ensure cat_col is categorical and num_col is numeric
                            if cat_col and num_col:
                                if not pd.api.types.is_numeric_dtype(df[num_col]):
                                    # Swap if needed
                                    cat_col, num_col = num_col, cat_col
                        elif len(terms) == 1:
                            # If only one term, assume it's categorical and we'll count
                            cat_col = matching_column(df, terms[0], prefer_categorical=True)
                        else:
                            # Default to first categorical and numerical columns
                            cat_cols = df.select_dtypes(include=['object', 'category']).columns
                            if not cat_cols.empty:
                                cat_col = cat_cols[0]
                            
                            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            if not num_cols.empty:
                                num_col = num_cols[0]
                        
                        st.info(f"Bar chart using category: {cat_col}, value: {num_col}")
                        
                        if cat_col:
                            if num_col and pd.api.types.is_numeric_dtype(df[num_col]):
                                # Aggregate data
                                agg_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                agg_data.plot.bar(ax=ax)
                                plt.title(f'Average {num_col} by {cat_col}')
                                plt.xlabel(cat_col)
                                plt.ylabel(f'Average {num_col}')
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                            else:
                                # Fall back to count if no numeric column
                                fig, ax = plt.subplots(figsize=(10, 6))
                                df[cat_col].value_counts().head(10).plot.bar(ax=ax)
                                plt.title(f'Count of {cat_col}')
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                        else:
                            st.warning("No suitable category column found for bar chart")
                    
                    # Scatter plot
                    elif chart_type == "scatter":
                        x_col = None
                        y_col = None
                        
                        # If we have at least two terms
                        if len(terms) >= 2:
                            # First term is x, second is y
                            x_col = matching_column(df, terms[0], prefer_categorical=False)
                            y_col = matching_column(df, terms[1], prefer_categorical=False)
                        else:
                            # Default to first two numeric columns
                            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            if len(num_cols) >= 2:
                                x_col = num_cols[0]
                                y_col = num_cols[1]
                        
                        st.info(f"Scatter plot using x: {x_col}, y: {y_col}")
                        
                        if x_col and y_col and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
                            plt.title(f'{x_col} vs {y_col}')
                            plt.xlabel(x_col)
                            plt.ylabel(y_col)
                            st.pyplot(fig)
                            st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                        else:
                            st.warning("No suitable numeric columns for scatter plot")
                    
                    # Pie chart
                    elif chart_type == "pie":
                        cat_col = None
                        
                        # If we have at least one term
                        if terms:
                            cat_col = matching_column(df, terms[0], prefer_categorical=True)
                        else:
                            # Default to first categorical column
                            cat_cols = df.select_dtypes(include=['object', 'category']).columns
                            if not cat_cols.empty:
                                cat_col = cat_cols[0]
                        
                        st.info(f"Pie chart using category: {cat_col}")
                        
                        if cat_col:
                            count_data = df[cat_col].value_counts().head(6)  # Limit to top 6 for readability
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax.pie(count_data, labels=count_data.index, autopct='%1.1f%%', startangle=90)
                            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                            plt.title(f'Distribution of {cat_col}')
                            st.pyplot(fig)
                            st.session_state.generated_charts[f"{chart_type}_{matching_column}"] = fig
                        else:
                            st.warning("No suitable categorical column found for pie chart")
                    
                    # Default case
                    else:
                        st.warning("Unrecognized command. Please use one of the suggested command formats.")
                
                except Exception as e:
                    st.error(f"Error creating visualization: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        if st.button("Generate Full Dashboard", type="primary"):
            st.session_state.show_dashboard = True
            st.rerun()  
        
    else:
        st.info("Please upload a data file first")

    update_created_charts_list()

st.markdown("---")
if st.button("Generate Comprehensive Dashboard", key="gen_dashboard"):
    if st.session_state.generated_charts:
        try:
            dashboard_path = generate_dashboard(st.session_state.generated_charts)
            st.success(f"✅ Dashboard generated and saved to: {dashboard_path}")
            
            # Display the saved dashboard
            dashboard_img = plt.imread(dashboard_path)
            st.image(dashboard_img, caption="Generated Dashboard", use_container_width=True)
        except Exception as e:
            st.error(f"Error generating dashboard: {str(e)}")
    else:
        st.warning("No charts generated yet. Create some visualizations first.")

# Show comprehensive dashboard
if st.session_state.show_dashboard and st.session_state.df is not None:
    st.markdown("---")
    create_dashboard(st.session_state.df)
