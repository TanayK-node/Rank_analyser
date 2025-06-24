import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Ranking Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stDownloadButton button {
        width: 100%;
    }
    .highlight {
        background-color: #fffacd;
        font-weight: bold;
    }
    .positive {
        color: green;
    }
    .negative {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

def style_negative_positive(val):
    if isinstance(val, (int, float)):
        color = 'green' if val >= 0 else 'red'
        return f'color: {color}'
    return ''

def main():
    st.title("📊 Ranking Analysis Tool")
    st.markdown("""
    A comprehensive tool for analyzing and ranking stocks based on multiple financial metrics.
    Upload your data, configure the analysis, and gain valuable insights.
    """)
    
    # Initialize session state
    if 'df_processed' not in st.session_state:
        st.session_state.df_processed = None
    if 'benchmark' not in st.session_state:
        st.session_state.benchmark = None
    
    # File upload section
    st.sidebar.header("1. Data Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Stock Data (CSV)", 
        type="csv",
        help="Upload your stock data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Data uploaded successfully!")
            
            
            # Get column types
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            non_numeric_cols = list(set(df.columns) - set(numeric_cols))
            
            name_col = st.sidebar.selectbox(
            "Company Name Column", 
            options=non_numeric_cols,
            index=non_numeric_cols.index("Name") if "Name" in non_numeric_cols else 0
        )

            
            # Sector/Industry analysis
            sector_col = None
            if len(non_numeric_cols) > 1:
                sector_col = st.sidebar.selectbox(
                    "Sector/Industry Column (Optional)", 
                    options=["None"] + non_numeric_cols,
                    index=0
                )
                if sector_col == "None":
                    sector_col = None
            
            # Filtering options
            with st.sidebar.expander("🔎 Filtering Options", expanded=True):
                filter_market_cap = st.checkbox("Filter by Market Cap", True)
                if filter_market_cap:
                    min_market_cap = st.number_input(
                        "Minimum Market Cap (Cr)", 
                        min_value=0, 
                        value=500
                    )
                
                filter_profit = st.checkbox("Filter by Profit", True)
                if filter_profit:
                    min_profit = st.number_input(
                        "Minimum Profit (Cr)", 
                        min_value=0, 
                        value=1
                    )
                
                filter_sales = st.checkbox("Filter by Sales", True)
                if filter_sales:
                    min_sales = st.number_input(
                        "Minimum Sales (Cr)", 
                        min_value=0, 
                        value=100
                    )
            
            # Metric configuration
            st.sidebar.header("2. Analysis Configuration")
            
            # Default metrics from Code 1
            default_metrics = [
                'QoQ Sales', 'Sales growth 3Years', 'Sales growth 5Years',
                'Sales growth', 'YOY Quarterly sales growth', '2Qoq Sales',
                'QoQ EPS growth', 'EPS growth 3Years', 'EPS growth 5Years',
                'Profit growth 3Years', 'Profit growth 5Years', 'Profit growth',
                'YOY Quarterly profit growth', 'QoQ Profits',
                'OPM', 'NPM last year', 'QoQ Op Profit growth', '2QoQ op profit', 'Operating profit growth',
                'Return on equity', 'Return on capital employed',
                'Average return on capital employed 5Years', 'Return on assets',
                'PEG Ratio', 'Debt to equity', 'Interest Coverage Ratio', 'Current ratio'
            ]
            
            # Let user select which columns to include
            selected_metrics = st.sidebar.multiselect(
                "Select Metrics for Analysis",
                options=numeric_cols,
                default=[m for m in default_metrics if m in numeric_cols],
                help="Choose which metrics to include in the ranking"
            )
            
            # Initialize metric categories with Code 1's structure
            if 'metric_categories' not in st.session_state: 
                st.session_state.metric_categories = {
                    'Sales': {
                        'weight': 0.7,
                        'metrics': [col for col in selected_metrics if any(x in col.lower() for x in ['sales'])]
                    },
                    'EPS': {
                        'weight': 1.2,
                        'metrics': [col for col in selected_metrics if 'eps' in col.lower()]
                    },
                    'Profit': {
                        'weight': 1.0,
                        'metrics': [col for col in selected_metrics if 'profit' in col.lower()]
                    },
                    'Operational': {
                        'weight': 0.9,
                        'metrics': [col for col in selected_metrics if any(x in col.lower() for x in ['opm', 'npm', 'operating'])]
                    },
                    'Returns': {
                        'weight': 1.1,
                        'metrics': [col for col in selected_metrics if 'return' in col.lower()]
                    },
                    'Efficiency': {
                        'weight': 1.2,
                        'metrics': [col for col in selected_metrics if any(x in col.lower() for x in ['ratio', 'debt', 'peg'])]
                    }
                }
            
            # Allow user to edit categories
            with st.sidebar.expander("⚙️ Metric Categories & Weights", expanded=True):
                categories = list(st.session_state.metric_categories.keys())
                new_categories = {}
                
                for cat in categories:
                    cols = st.multiselect(
                        f"{cat} Metrics",
                        options=selected_metrics,
                        default=st.session_state.metric_categories[cat]['metrics'],
                        key=f"metrics_{cat}"
                    )
                    
                    weight = st.slider(
                        f"{cat} Weight",
                        min_value=0.1,
                        max_value=2.0,
                        value=st.session_state.metric_categories[cat]['weight'],
                        step=0.1,
                        key=f"weight_{cat}"
                    )
                    
                    new_categories[cat] = {
                        'weight': weight,
                        'metrics': cols
                    }
                
                # Add option to add new category
                new_cat_name = st.text_input("➕ Add New Category (press enter to add)")
                if new_cat_name and new_cat_name not in new_categories:
                    new_cat_metrics = st.multiselect(
                        f"{new_cat_name} Metrics",
                        options=[m for m in selected_metrics if not any(m in cat['metrics'] for cat in new_categories.values())],
                        key=f"new_metrics_{new_cat_name}"
                    )
                    if new_cat_metrics:
                        new_cat_weight = st.slider(
                            f"{new_cat_name} Weight",
                            min_value=0.1,
                            max_value=2.0,
                            value=1.0,
                            step=0.1,
                            key=f"new_weight_{new_cat_name}"
                        )
                        new_categories[new_cat_name] = {
                            'weight': new_cat_weight,
                            'metrics': new_cat_metrics
                        }
                
                st.session_state.metric_categories = new_categories
            
             #Benchmark selection
            st.sidebar.subheader("📊 Benchmark Options")
            benchmark = st.sidebar.selectbox(
                "Select Benchmark Stock",
                options=["None"] + df[name_col].tolist(),
                index=0
            )
            if benchmark != "None":
                st.session_state.benchmark = benchmark 
            
            # Process data when user clicks the button
            # Process data when user clicks the button
            if st.sidebar.button("🚀 Calculate Rankings", type="primary", help="Run the analysis with current settings"):
                with st.spinner("Crunching numbers... Please wait"):
                    try:
                        # Make a copy of the dataframe
                        df_processed = df.copy()
                        
                        # Convert numeric columns
                        for col in selected_metrics:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                        
                        # Apply filters
                        if filter_market_cap and 'Market Capitalization' in df_processed.columns:
                            df_processed = df_processed[df_processed['Market Capitalization'] > min_market_cap]
                        if filter_profit and 'Net profit' in df_processed.columns:
                            df_processed = df_processed[df_processed['Net profit'] > min_profit]
                        if filter_sales and 'Sales' in df_processed.columns:
                            df_processed = df_processed[df_processed['Sales'] > min_sales]
                        
                        df_processed[selected_metrics] = df_processed[selected_metrics].apply(
                        lambda col: col.where(pd.notnull(col), np.nan)
                        )
                        
                        # Calculate ranks for each metric
                        for cat in st.session_state.metric_categories.values():
                            for col in cat['metrics']:
                                if col in df_processed.columns:
                                    # For debt/peg ratios, lower is better (ascending=True)
                                    ascending = True if any(term in col.lower() for term in ['debt', 'peg', 'ratio']) else False
                                    df_processed[f'Rank_{col}'] = df_processed[col].rank(ascending=ascending)
                        
                        # Calculate weighted average ranks
                        weighted_ranks = []
                        for cat in st.session_state.metric_categories.values():
                            cat_ranks = [f'Rank_{col}' for col in cat['metrics'] if f'Rank_{col}' in df_processed.columns]
                            if cat_ranks:
                                avg_cat_rank = df_processed[cat_ranks].mean(axis=1)
                                weighted_ranks.append(avg_cat_rank * cat['weight'])
                        
                        if weighted_ranks:
                            # Combine all weighted ranks and normalize
                            df_processed['Weighted_Avg_Rank'] = sum(weighted_ranks)
                            max_rank = df_processed['Weighted_Avg_Rank'].max()
                            df_processed['Score'] = df_processed['Weighted_Avg_Rank'].apply(
                                lambda x: round((1 - x / max_rank) * 1000, 2))
                            
                            # Calculate percentile ranks
                            df_processed['Percentile'] = df_processed['Score'].rank(pct=True).round(2) * 100
                            
                            # Final sorting
                            df_processed = df_processed.sort_values(by='Score', ascending=False)
                            
                            # Store in session state
                            st.session_state.df_processed = df_processed
                            
                            # Collapse the sidebar
                            st.session_state.sidebar_expanded = False
                            st.sidebar.success("Analysis complete! (Clsoe Sidebar to view results)")
                            
                            st.success("Analysis complete!")
                        else:
                            st.error("No valid metrics selected for analysis.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            # Display results if available
            if st.session_state.df_processed is not None:
                df_processed = st.session_state.df_processed
                
                 # Add interactive score filter
                st.sidebar.header("🔍 Results Filtering")
                min_score = st.sidebar.slider(
                    "Minimum Score Filter", 
                    min_value=0,
                    max_value=1000,
                    value=0,
                    help="Filter companies by minimum score"
                )
                df_processed = df_processed[df_processed['Score'] >= min_score]
                
                # Summary stats
                st.header("📈 Analysis Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Companies Analyzed", len(df_processed))
                with col2:
                    top_score = df_processed['Score'].max()
                    st.metric("Top Score", round(top_score, 2))
                with col3:
                    avg_score = df_processed['Score'].mean()
                    st.metric("Average Score", round(avg_score, 2))
                
               # Top companies section
                st.subheader("🏆 Top Ranked Companies")
                
                # Add option to select number of companies to display
                num_companies = st.slider(
                    "Number of companies to display", 
                    min_value=5, 
                    max_value=100, 
                    value=20,  # Default value
                    step=5
                )
                
               # Create display columns
                display_cols = [name_col, 'Score', 'Percentile'] + ['Current Price'] + selected_metrics
                display_cols = [col for col in display_cols if col in df_processed.columns]

                # Show selected number of top companies
                top_companies = df_processed[display_cols].head(num_companies)

                # Create a format dictionary for all numeric columns (excluding name_col)
                format_dict = {
                    'Score': '{:.1f}',
                    'Percentile': '{:.1f}%'
                }

                # Add formatting for other numeric columns (assuming they should have 1 decimal place)
                numeric_cols = top_companies.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    if col not in format_dict and col != name_col:
                        format_dict[col] = '{:.1f}'

                # Format the display
                styled_df = top_companies.style.format(format_dict).applymap(style_negative_positive, subset=selected_metrics)

                # Display without index (serial numbers)
                st.dataframe(
                    styled_df,
                    height=600,
                    use_container_width=True,
                    hide_index=True  # This removes the serial numbers
                )
                
                # Highlight benchmark if selected
                if st.session_state.benchmark and st.session_state.benchmark in df_processed[name_col].values:
                    highlight_idx = df_processed[df_processed[name_col] == st.session_state.benchmark].index[0]
                    st.info(f"🌟 Benchmark: {st.session_state.benchmark} (Rank: {list(df_processed.index).index(highlight_idx) + 1}, Score: {df_processed.loc[highlight_idx
                            , 'Score']:.2f})")
                
                # Format the display
                styled_df = top_companies.style.format({
                    'Score': '{:.2f}',
                    'Percentile': '{:.1f}%'
                }).applymap(style_negative_positive, subset=selected_metrics)
        
                
                # Visualizations
                st.subheader("📊 Visual Analysis")
                
                # Score distribution
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Risk Analysis", "Score Composition", "Score Distribution", "Benchmark Comparison","Sector Analysis"])
                
                with tab1:                    
                    if 'Return on equity' in df_processed.columns and 'Debt to equity' in df_processed.columns:
                        fig = px.scatter(
                            df_processed,
                            x='Debt to equity',  # Risk (x-axis)
                            y='Return on equity',  # Reward (y-axis)
                            color='Score',
                            size='Market Capitalization',
                            hover_name=name_col,
                            title="Risk-Reward Analysis (Size=Market Cap, Color=Score)",
                            labels={
                                'Debt to equity': 'Debt/Equity (Risk →)',
                                'Return on equity': 'ROE (Reward ↑)'
                            },
                            color_continuous_scale='RdYlGn',  # Red-Yellow-Green scale
                            size_max=20
                        )
                        
                        # Add quadrant lines and annotations
                        median_debt = df_processed['Debt to equity'].median()
                        median_roe = df_processed['Return on equity'].median()
                        
                        fig.add_hline(y=median_roe, line_dash="dot", line_color="gray")
                        fig.add_vline(x=median_debt, line_dash="dot", line_color="gray")
                        
                        # Highlight benchmark if available
                        if st.session_state.benchmark and st.session_state.benchmark in df_processed[name_col].values:
                            fig.add_trace(
                                px.scatter(
                                    df_processed[df_processed[name_col] == st.session_state.benchmark],
                                    x='Debt to equity',
                                    y='Return on equity',
                                    text=name_col
                                ).update_traces(
                                    marker=dict(color='black', size=20, symbol='star'),
                                    textposition='top center'
                                ).data[0]
                            )
                        
                        # Add quadrant labels
                        fig.add_annotation(x=0.1, y=median_roe*1.5, text="High Reward/Low Risk", showarrow=False)
                        fig.add_annotation(x=median_debt*1.5, y=median_roe*1.5, text="High Reward/High Risk", showarrow=False)
                        fig.add_annotation(x=0.1, y=median_roe*0.5, text="Low Reward/Low Risk", showarrow=False)
                        fig.add_annotation(x=median_debt*1.5, y=median_roe*0.5, text="Low Reward/High Risk", showarrow=False)
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Required metrics (ROE/Debt-to-Equity) not available for risk-reward analysis")
                
                with tab2:
                    st.subheader("Score Composition: Top 8 Companies")

                    # Get top 8 companies by Score
                    top_companies = df_processed.sort_values(by='Score', ascending=False).head(8)

                    # Create a structure to hold contributions
                    all_contributions = []

                    total_weight = sum(cat['weight'] for cat in st.session_state.metric_categories.values())

                    for _, row in top_companies.iterrows():
                        company_name = row[name_col]
                        company_contributions = {'Company': company_name}

                        for cat_name, cat_data in st.session_state.metric_categories.items():
                            if cat_data['metrics']:
                                # Get relevant rank columns
                                rank_cols = [f'Rank_{col}' for col in cat_data['metrics'] if f'Rank_{col}' in df_processed.columns]
                                if rank_cols:
                                    avg_rank = row[rank_cols].mean()
                                    contribution = (cat_data['weight'] * (1 - (avg_rank / len(df_processed)))) / total_weight
                                    company_contributions[cat_name] = contribution * 100  # as percentage
                                else:
                                    company_contributions[cat_name] = 0
                            else:
                                company_contributions[cat_name] = 0

                        all_contributions.append(company_contributions)

                    # Convert to DataFrame for plotting
                    df_contrib = pd.DataFrame(all_contributions)

                    # Melt the DataFrame for stacked bar chart
                    df_melted = df_contrib.melt(id_vars='Company', var_name='Category', value_name='Contribution')

                    # Sort by total score
                    df_melted['Company'] = pd.Categorical(df_melted['Company'], 
                                                        categories=top_companies[name_col].tolist(),
                                                        ordered=True)

                    # Plot stacked bar chart
                    fig = px.bar(
                        df_melted,
                        x='Company',
                        y='Contribution',
                        color='Category',
                        title="Category-wise Score Composition of Top 8 Companies",
                        text_auto='.1f',
                        labels={'Contribution': 'Contribution (%)'},
                        height=500
                    )

                    fig.update_layout(
                        barmode='stack',
                        xaxis_title="Company",
                        yaxis_title="Contribution (%)",
                        legend_title="Metric Category"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                
                # Create a risk-reward scatter plot
                with tab3:
                    fig = px.histogram(df_processed, x='Score', nbins=20, 
                                      title="Distribution of Scores")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    if st.session_state.benchmark and st.session_state.benchmark in df_processed[name_col].values:
                        benchmark_data = df_processed[df_processed[name_col] == st.session_state.benchmark].iloc[0]
                        
                        # Compare against sector average if available
                        if sector_col:
                            sector = benchmark_data[sector_col]
                            sector_avg = df_processed[df_processed[sector_col] == sector][selected_metrics].mean()
                            
                            comparison = pd.DataFrame({
                                'Metric': selected_metrics,
                                'Benchmark': benchmark_data[selected_metrics].values,
                                'Sector Average': sector_avg.values
                            })
                            
                            fig = px.bar(comparison.melt(id_vars='Metric'), 
                                        x='Metric', y='value', color='variable',
                                        barmode='group', 
                                        title=f"{st.session_state.benchmark} vs Sector Average")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.write(f"Detailed metrics for {st.session_state.benchmark}:")
                            st.dataframe(benchmark_data[selected_metrics])
                    else:
                        st.warning("No benchmark selected or benchmark not in filtered data")
                
                with tab5:
                    if sector_col:
                        sector_avg = df_processed.groupby(sector_col)['Score'].mean().sort_values(ascending=False)
                        fig = px.bar(sector_avg, x=sector_avg.index, y='Score',
                                    title="Average Score by Sector")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No sector/industry column selected")
                    
                    st.subheader("🧩 Score Composition Analysis")
                    
                   
                # Detailed metric analysis
                st.subheader("🔍 Metric Performance")
                
                for cat_name, cat_data in st.session_state.metric_categories.items():
                    with st.expander(f"{cat_name} Metrics (Weight: {cat_data['weight']})"):
                        if cat_data['metrics']:
                            # Show top and bottom performers for each metric
                            for metric in cat_data['metrics']:
                                if metric in df_processed.columns:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Top 5 {metric}**")
                                        top_5 = df_processed.nlargest(5, metric)[[name_col, metric]]
                                        st.dataframe(top_5.style.format({metric: '{:.2f}'}))
                                    with col2:
                                        st.write(f"**Bottom 5 {metric}**")
                                        bottom_5 = df_processed.nsmallest(5, metric)[[name_col, metric]]
                                        st.dataframe(bottom_5.style.format({metric: '{:.2f}'}))
                        else:
                            st.warning("No metrics selected for this category")
                
                # Download section
                st.subheader("💾 Download Results")
                csv = df_processed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Analysis (CSV)",
                    data=csv,
                    file_name="stock_analysis_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("👈 Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()