"""
SupportIQ Web - Streamlit Application

This is the main application for the Customer Support Analytics Dashboard.
It provides interactive visualizations and analytics for support ticket data.

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# Import path manager for correct file paths
from customer_support_analytics.core.path import path_manager

# Set page config
st.set_page_config(
    page_title="SupportIQ Web",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data from CSV files
@st.cache_data
def load_csv_data():
    data = {
        'daily_metrics': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'daily_metrics.csv'),
        'issue_type_metrics': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'issue_type_metrics.csv'),
        'agent_performance': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'agent_performance.csv'),
        'priority_metrics': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'priority_metrics.csv'),
        'channel_metrics': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'channel_metrics.csv'),
        'tickets': pd.read_csv(path_manager.PROCESSED_DATA_DIR / 'tickets_cleaned.csv')
    }
    # Convert date columns
    if 'created_date' in data['daily_metrics'].columns:
        data['daily_metrics']['created_date'] = pd.to_datetime(data['daily_metrics']['created_date'])
    if 'created_date' in data['tickets'].columns:
        data['tickets']['created_date'] = pd.to_datetime(data['tickets']['created_date'])
    if 'ticket_created_at' in data['tickets'].columns:
        data['tickets']['ticket_created_at'] = pd.to_datetime(data['tickets']['ticket_created_at'])
    return data

data = load_csv_data()

# Sidebar navigation
st.sidebar.title("SupportIQ Web")
page = st.sidebar.radio("Navigation", [
    "Application Overview",
    "Executive Dashboard",
    "Ticket Analysis",
    "Agent Performance",
    "Issue Analytics",
    "SLA Monitoring",
    "CSAT Insights",
    "Trend Analysis",
    "Experiment Analysis",
    "Time Series Forecasting",
    "CATE Analysis"
])

# Application Overview
if page == "Application Overview":
    st.title("📊 Application Overview")
    st.markdown("""
    ## Customer Support Analytics Platform
    
    Welcome to the **Customer Support Analytics Platform** — a comprehensive intelligence 
    system designed to transform raw support ticket data into actionable business insights.
    
    This platform enables data-driven decision-making for customer support operations through 
    interactive visualizations, statistical analysis, and machine learning-powered predictions.
    """)
    
    # Platform Purpose
    st.markdown("---")
    st.subheader("🎯 Platform Purpose")
    
    st.markdown("""
    **What This Platform Does:**
    
    This application serves as a centralized hub for understanding and optimizing your 
    customer support operations. It processes historical support data to reveal patterns, 
    identify problems, and recommend actions.
    
    **Core Value Proposition:**
    - **Visibility**: See comprehensive metrics across your entire support operation
    - **Insights**: Understand why metrics change and what drives customer satisfaction
    - **Predictions**: Forecast future demand and plan resources accordingly
    - **Validation**: Test changes with rigorous statistical methods before full rollout
    """)
    
    # Target Users
    st.markdown("---")
    st.subheader("👥 Who Uses This Platform")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📋 Support Managers**
        
        *Daily operational oversight*
        
        - Monitor team performance in real-time
        - Identify bottlenecks before they escalate
        - Allocate resources efficiently across queues
        - Coach agents based on data-driven insights
        """)
    with col2:
        st.markdown("""
        **📈 Operations Leaders**
        
        *Strategic planning and optimization*
        
        - Track SLA compliance trends
        - Optimize support processes
        - Forecast staffing needs
        - Identify training priorities
        """)
    with col3:
        st.markdown("""
        **🔬 Data Scientists**
        
        *Advanced analytics and experimentation*
        
        - Design and analyze A/B tests
        - Build causal inference models
        - Estimate treatment effects
        - Develop predictive algorithms
        """)
    
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("""
        **🚀 Product Teams**
        
        *Impact validation and customer understanding*
        
        - Measure impact of product changes
        - Understand customer pain points
        - Prioritize features by support burden
        - Validate tool investments
        """)
    with col5:
        st.markdown("""
        **🏢 Executive Leadership**
        
        *High-level strategic decisions*
        
        - Executive KPI dashboards
        - Trend analysis for planning
        - Comparative performance views
        - ROI validation for initiatives
        """)
    
    # Core Capabilities
    st.markdown("---")
    st.subheader("⚡ Core Capabilities")
    
    st.markdown("""
    The platform provides six integrated analytical capabilities:
    """)
    
    capabilities = {
        '📊 Operational Monitoring': {
            'description': 'Real-time visibility into support health',
            'features': ['KPI tracking with trend indicators', 'Daily/weekly/monthly views', 'Automated anomaly alerts'],
            'benefit': 'Know your support health at a glance'
        },
        '📈 Performance Analytics': {
            'description': 'Deep understanding of support operations',
            'features': ['Agent performance rankings', 'Issue type analysis', 'Channel comparison', 'Priority-based metrics'],
            'benefit': 'Identify improvement opportunities'
        },
        '✅ Quality Assurance': {
            'description': 'Maintain and improve service quality',
            'features': ['SLA breach tracking', 'CSAT trend analysis', 'Anomaly detection', 'Root cause identification'],
            'benefit': 'Deliver consistent, excellent service'
        },
        '🧪 Experimentation': {
            'description': 'Validate changes with statistical rigor',
            'features': ['A/B testing framework', 'Sample size calculator', 'Statistical significance testing', 'Effect size analysis'],
            'benefit': 'Make confident, data-backed decisions'
        },
        '📉 Predictive Analytics': {
            'description': 'Anticipate future support needs',
            'features': ['Volume forecasting', 'Seasonality detection', 'Trend extrapolation', 'Capacity planning'],
            'benefit': 'Proactive resource allocation'
        },
        '🧠 Causal Inference': {
            'description': 'Understand true cause-and-effect relationships',
            'features': ['Treatment effect estimation', 'Heterogeneous effects (CATE)', 'Meta-learner approaches', 'Sensitivity analysis'],
            'benefit': 'Target interventions effectively'
        }
    }
    
    for cap_title, cap_data in capabilities.items():
        with st.expander(f"{cap_title} — {cap_data['description']}"):
            st.markdown(f"**Features:**")
            for feature in cap_data['features']:
                st.markdown(f"  - {feature}")
            st.markdown(f"**Benefit:** {cap_data['benefit']}")
    
    # Architectural Structure
    st.markdown("---")
    st.subheader("🏗️ How The Platform Works")
    
    st.markdown("""
    **Three-Layer Architecture:**
    
    The platform is built on a three-layer architecture that processes raw data into actionable insights:
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1️⃣ Data Layer**
        
        *Raw data → Structured datasets*
        
        - Loads CSV files from storage
        - Converts dates and data types
        - Caches data for performance
        - Provides data to all pages
        """)
    with col2:
        st.markdown("""
        **2️⃣ Computation Layer**
        
        *Data → Metrics & Statistics*
        
        - Aggregates ticket data
        - Calculates KPIs and trends
        - Runs statistical tests
        - Generates forecasts
        """)
    with col3:
        st.markdown("""
        **3️⃣ Presentation Layer**
        
        *Metrics → Visual Insights*
        
        - Renders interactive charts
        - Displays KPI cards with deltas
        - Shows sortable data tables
        - Provides context and guidance
        """)
    
    # Data Sources
    st.markdown("---")
    st.subheader("📁 Data Sources")
    
    st.markdown("""
    The platform processes data from six integrated sources:
    """)
    
    data_sources = [
        ('daily_metrics.csv', 'Time-series aggregated metrics', 'Daily rollups of tickets, CSAT, resolution time, and SLA metrics over the last 30 days'),
        ('tickets_cleaned.csv', 'Individual ticket records', 'Detailed ticket-level data including channel, priority, agent, and customer satisfaction scores'),
        ('agent_performance.csv', 'Agent-level aggregations', 'Summary metrics per agent including ticket volume, response times, and CSAT ratings'),
        ('issue_type_metrics.csv', 'Issue category breakdowns', 'Metrics segmented by type of issue (e.g., billing, technical, shipping)'),
        ('priority_metrics.csv', 'Priority-based aggregations', 'Metrics segmented by priority level (critical, high, medium, low)'),
        ('channel_metrics.csv', 'Channel comparison data', 'Metrics segmented by support channel (email, chat, phone, social)')
    ]
    
    for source_name, source_type, source_desc in data_sources:
        st.markdown(f"- **{source_name}** ({source_type}): {source_desc}")
    
    # Metrics Framework
    st.markdown("---")
    st.subheader("📏 Metrics Framework")
    
    st.markdown("""
    The platform tracks four categories of metrics, each with specific business meaning:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **📦 Volume Metrics**
        
        *Understanding demand*
        
        - **Total Tickets**: Raw demand indicator
        - **Ticket Distribution**: Where issues come from
        - **Peak Hours**: When demand is highest
        
        **Why it matters**: Volume drives capacity planning and staffing decisions.
        """)
        st.markdown("""
        **⚡ Efficiency Metrics**
        
        *Measuring operational performance*
        
        - **First Response Time**: Initial acknowledgment speed
        - **Resolution Time**: Total time to solve
        - **Solved Rate**: Closure effectiveness
        
        **Why it matters**: Efficiency impacts both costs and customer satisfaction.
        """)
    with col2:
        st.markdown("""
        **😊 Quality Metrics**
        
        *Measuring customer satisfaction*
        
        - **CSAT Score**: Direct customer rating (1-5)
        - **Solved Rate**: Successful closures
        - **First Contact Resolution**: Issues resolved immediately
        
        **Why it matters**: Quality metrics predict customer loyalty and retention.
        """)
        st.markdown("""
        **🚨 Risk Metrics**
        
        *Identifying problems early*
        
        - **SLA Breach Rate**: Service commitment failures
        - **Escalation Rate**: Issues requiring manager intervention
        - **Low CSAT Rate**: Dissatisfied customers
        
        **Why it matters**: Risk metrics indicate potential customer churn and operational issues.
        """)
    
    # Statistical Methods
    st.markdown("---")
    st.subheader("🔬 Statistical Analysis Framework")
    
    st.markdown("""
    **Hypothesis Testing Approach:**
    
    When you want to validate a change (e.g., "Will a new tool improve agent productivity?"), 
    the platform uses statistical hypothesis testing to determine if observed differences 
    are real or due to chance.
    
    | Method | Purpose | Business Question It Answers |
    |--------|---------|------------------------------|
    | **t-test** | Compare means between two groups | "Is the variant better than control?" |
    | **Chi-square test** | Compare proportions/rates | "Did the breach rate change after our fix?" |
    | **Mann-Whitney U** | Non-parametric comparison | "Is the improvement real, even with outliers?" |
    | **ANOVA** | Compare multiple groups | "Which channel has the best CSAT?" |
    """)
    
    st.markdown("""
    **Interpretation Guide:**
    
    - **p-value < 0.05**: Results are statistically significant (95% confidence)
    - **Effect size (Cohen's d)**: Measures practical significance
      - Small (0.2): Noticeable but may not be business-relevant
      - Medium (0.5): Meaningful improvement
      - Large (0.8): Transformative impact
    """)
    
    # Causal Inference
    st.markdown("---")
    st.subheader("🧠 Causal Inference Framework")
    
    st.markdown("""
    **Moving Beyond Correlation:**
    
    While correlation shows that two metrics move together, **causation** shows that 
    one metric directly causes changes in another. Causal inference methods help 
    answer: "Did our intervention actually cause the improvement, or would it have 
    happened anyway?"
    
    | Method | Business Question It Answers |
    |--------|------------------------------|
    | **Difference-in-Differences (DiD)** | "Did the policy change actually improve CSAT, or was it just seasonal?" |
    | **Propensity Score Matching (PSM)** | "What would have happened to treated customers without the intervention?" |
    | **Meta-Learners (CATE)** | "Which specific customer segments benefit most from our intervention?" |
    """)
    
    st.markdown("""
    **Practical Application:**
    
    Causal inference is critical when:
    - Evaluating expensive interventions (training programs, new tools)
    - Making resource allocation decisions
    - Validating that changes you made actually caused improvements
    - Identifying which customers respond best to specific treatments
    """)
    
    # Problems Addressed
    st.markdown("---")
    st.subheader("💡 Business Problems This Platform Solves")
    
    problems = [
        {
            'title': '📊 Support Performance Visibility',
            'problem': 'How do I know if our support operation is healthy?',
            'solution': 'Executive Dashboard provides at-a-glance KPI overview with trend indicators showing daily/weekly changes.',
            'action': 'Check the Executive Dashboard first when assessing support health.'
        },
        {
            'title': '🔍 Root Cause Identification',
            'problem': 'Why did our CSAT drop last week?',
            'solution': 'Multi-dimensional filtering and distribution visualizations reveal correlations with volume spikes, staffing changes, or specific issue types.',
            'action': 'Use Ticket Analysis with filters to drill down by channel, priority, and issue type.'
        },
        {
            'title': '🧪 Experiment Validation',
            'problem': 'Will the new tool actually improve agent productivity?',
            'solution': 'Sample size calculator prevents under-powered tests; statistical testing framework validates results with significance and effect size.',
            'action': 'Use Experiment Analysis to design tests before launching and analyze results after.'
        },
        {
            'title': '🎯 Heterogeneous Impact Understanding',
            'problem': 'Which customers benefit most from proactive support?',
            'solution': 'CATE meta-learners estimate treatment effects per subgroup, enabling targeted resource allocation.',
            'action': 'Use CATE Analysis to identify high-value intervention targets.'
        },
        {
            'title': '📈 Proactive Resource Planning',
            'problem': 'How do I plan staffing for next month?',
            'solution': 'Time series forecasting predicts future ticket volume with confidence intervals, accounting for seasonality and trends.',
            'action': 'Use Time Series Forecasting to generate staffing recommendations.'
        }
    ]
    
    for problem in problems:
        with st.expander(f"{problem['title']}"):
            st.markdown(f"**Problem**: {problem['problem']}")
            st.markdown(f"**Solution**: {problem['solution']}")
            st.markdown(f"**How to Use**: {problem['action']}")
    
    # Decision Support Framework
    st.markdown("---")
    st.subheader("📋 Decision Support Framework")
    
    st.markdown("""
    The platform transforms from a **reporting tool** to a **decision support system** 
    by answering four progressive questions:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        **1️⃣ What happened?**
        
        *Descriptive Analytics*
        
        - Current metric values
        - Historical comparisons
        - Trend indicators
        
        **Pages**: Executive Dashboard, Ticket Analysis
        """)
    with col2:
        st.markdown("""
        **2️⃣ Why did it happen?**
        
        *Diagnostic Analytics*
        
        - Root cause analysis
        - Correlation analysis
        - Segment comparisons
        
        **Pages**: Issue Analytics, Agent Performance
        """)
    with col3:
        st.markdown("""
        **3️⃣ What will happen?**
        
        *Predictive Analytics*
        
        - Forecasting
        - Anomaly detection
        - Capacity planning
        
        **Pages**: Trend Analysis, Time Series Forecasting
        """)
    with col4:
        st.markdown("""
        **4️⃣ What should we do?**
        
        *Prescriptive Analytics*
        
        - Action recommendations
        - Experiment validation
        - Targeted interventions
        
        **Pages**: Experiment Analysis, CATE Analysis
        """)
    
    # Navigation Guide
    st.markdown("---")
    st.subheader("🗺️ Page Navigation Guide")
    
    st.markdown("""
    Use this guide to find the right page for your question:
    """)
    
    page_guide = [
        ('Executive Dashboard', 'High-level KPI overview', 'What is our current support health?'),
        ('Ticket Analysis', 'Detailed ticket drill-down', 'Show me specific tickets matching criteria'),
        ('Agent Performance', 'Team member evaluation', 'Who are our top performers? Who needs coaching?'),
        ('Issue Analytics', 'Issue type patterns', 'Which issue types cause the most problems?'),
        ('SLA Monitoring', 'SLA compliance tracking', 'Are we meeting our service commitments?'),
        ('CSAT Insights', 'Satisfaction analysis', 'How satisfied are our customers? What drives CSAT?'),
        ('Trend Analysis', 'Time-based patterns', 'When do we see peak demand? Any patterns?'),
        ('Experiment Analysis', 'A/B testing framework', 'Will this change actually help? Is it statistically significant?'),
        ('Time Series Forecasting', 'Predictive analytics', 'What will our volume be next week?'),
        ('CATE Analysis', 'Causal inference', 'Which customers respond best to interventions?')
    ]
    
    guide_df = pd.DataFrame(page_guide, columns=['Page', 'Purpose', 'Key Question'])
    st.dataframe(guide_df, use_container_width=True, hide_index=True)
    
    st.info("💡 **Getting Started**: If you're new to the platform, start with the Executive Dashboard for an overview, then drill into specific areas based on your questions.")

# Executive Dashboard
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown("### Key Performance Indicators (KPIs)")
    st.markdown("""
    This dashboard provides a high-level overview of your customer support operations.
    Use these metrics to monitor overall health and identify critical areas for improvement.
    """)
    
    # Load metrics
    daily_metrics = data['daily_metrics'].sort_values('created_date', ascending=False).head(30)
    issue_metrics = data['issue_type_metrics']
    priority_metrics = data['priority_metrics']
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Tickets
    total_tickets = daily_metrics['total_tickets'].sum()
    today_tickets = daily_metrics['total_tickets'].iloc[0]
    prev_day_tickets = daily_metrics['total_tickets'].iloc[1]
    ticket_delta = today_tickets - prev_day_tickets
    col1.metric("Total Tickets (30 days)", f"{total_tickets:,}", delta=f"{ticket_delta:+} vs prev day")
    with col1:
        st.markdown("""
        **Formula**: Sum of all tickets created in the period  
        **Interpretation**: High volume may indicate increased customer demand or service issues.  
        **Suggestion**: If trending upward, consider scaling support capacity or implementing self-service options.
        """)
    
    # Average CSAT
    avg_csat = daily_metrics['avg_csat'].mean()
    csat_delta = avg_csat - daily_metrics['avg_csat'].iloc[1]
    col2.metric("Average CSAT", f"{avg_csat:.2f}", delta=f"{csat_delta:+.2f}")
    with col2:
        st.markdown("""
        **Formula**: (Sum of CSAT scores) / (Number of responses)  
        **Interpretation**: Scores range 1-5. >4.5 = Excellent, <3.5 = Needs attention.  
        **Suggestion**: Low CSAT may indicate training gaps or process inefficiencies. Review low-scoring interactions.
        """)
    
    # Solved Rate
    avg_solved_rate = daily_metrics['solved_rate'].mean()
    solved_rate_delta = avg_solved_rate - daily_metrics['solved_rate'].iloc[1]
    col3.metric("Solved Rate", f"{avg_solved_rate:.1f}%", delta=f"{solved_rate_delta:+.1f}%")
    with col3:
        st.markdown("""
        **Formula**: (Solved tickets) / (Total tickets) × 100  
        **Interpretation**: High rates indicate efficient resolution processes.  
        **Suggestion**: If declining, investigate bottlenecks in escalation paths or agent workload.
        """)
    
    # SLA Breach Rate
    avg_sla_breach = daily_metrics['sla_breach_rate'].mean()
    sla_delta = avg_sla_breach - daily_metrics['sla_breach_rate'].iloc[1]
    col4.metric("SLA Breach Rate", f"{avg_sla_breach:.1f}%", delta=f"{sla_delta:+.1f}%")
    with col4:
        st.markdown("""
        **Formula**: (SLA breaches) / (Total tickets) × 100  
        **Interpretation**: Target <5% for most support operations.  
        **Suggestion**: Focus on reducing breaches for high-priority tickets first.
        """)
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Ticket Volume Trend")
        fig = px.line(daily_metrics, x='created_date', y='total_tickets', title='Daily Ticket Volume')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: Identify patterns like weekly peaks (often Mondays) or sudden spikes.  
        **Action**: Schedule additional agents during peak hours. Consider implementing chatbots for predictable high-volume periods.
        """)
    
    with col2:
        st.subheader("😊 CSAT Trend")
        fig = px.line(daily_metrics, x='created_date', y='avg_csat', title='Daily CSAT Score')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: Look for correlation with ticket volume spikes. Low CSAT following high volume may indicate burnout.  
        **Action**: Monitor CSAT closely after major incidents or releases. Provide support resources during stress periods.
        """)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Tickets by Issue Type")
        fig = px.pie(issue_metrics, values='total_tickets', names='issue_type', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: The largest segment indicates where most customer effort is spent.  
        **Action**: If one issue type dominates, consider product improvements, knowledge base articles, or dedicated support queues.
        """)
    
    with col2:
        st.subheader("🚨 Tickets by Priority")
        fig = px.bar(priority_metrics, x='priority', y='total_tickets', color='priority')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: Healthy distribution typically has more low/medium tickets than critical/high.  
        **Action**: High critical volume may indicate systemic issues. Review escalation criteria and SLAs.
        """)

# Ticket Analysis
elif page == "Ticket Analysis":
    st.title("🎫 Ticket Analysis")
    st.markdown("""
    Detailed analysis of ticket characteristics and performance. Use filters to drill down into specific segments.
    """)
    
    # Filters
    tickets = data['tickets']
    channels = tickets['channel'].unique().tolist()
    priorities = tickets['priority'].unique().tolist()
    issue_types = tickets['issue_type'].unique().tolist()
    
    col1, col2, col3 = st.columns(3)
    selected_channel = col1.selectbox("Channel", ['All'] + channels)
    selected_priority = col2.selectbox("Priority", ['All'] + priorities)
    selected_issue = col3.selectbox("Issue Type", ['All'] + issue_types)
    
    # Filter data
    filtered_tickets = tickets.copy()
    if selected_channel != 'All':
        filtered_tickets = filtered_tickets[filtered_tickets['channel'] == selected_channel]
    if selected_priority != 'All':
        filtered_tickets = filtered_tickets[filtered_tickets['priority'] == selected_priority]
    if selected_issue != 'All':
        filtered_tickets = filtered_tickets[filtered_tickets['issue_type'] == selected_issue]
    
    # Stats
    st.subheader("📈 Ticket Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_filtered = len(filtered_tickets)
    col1.metric("Total Tickets", total_filtered)
    with col1:
        st.markdown("**Filtered count** based on your selection criteria")
    
    avg_resolution = filtered_tickets['resolution_time_min'].mean()
    col2.metric("Avg Resolution Time", f"{avg_resolution:.0f} min")
    with col2:
        st.markdown("""
        **Formula**: Total resolution time / Tickets solved  
        **Benchmark**: Target <60 min for high priority, <4 hours for medium
        """)
    
    avg_response = filtered_tickets['first_response_time_min'].mean()
    col3.metric("Avg First Response", f"{avg_response:.0f} min")
    with col3:
        st.markdown("""
        **Formula**: Sum of first response times / Total tickets  
        **Impact**: First response is the strongest predictor of CSAT
        """)
    
    avg_csat = filtered_tickets['csat_score'].mean()
    col4.metric("Avg CSAT", f"{avg_csat:.2f}")
    with col4:
        st.markdown("""
        **Interpretation**: Direct measure of customer satisfaction  
        **Tip**: Filter by channel to identify best/worst performing channels
        """)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏱️ Resolution Time Distribution")
        fig = px.histogram(filtered_tickets, x='resolution_time_min', nbins=50, title='Resolution Time (minutes)')
        st.plotly_chart(fig, use_container_width=True)
        p90_resolution = np.percentile(filtered_tickets['resolution_time_min'].dropna(), 90)
        st.markdown(f"""
        **Analysis**: Right-skewed distribution is normal. Most tickets resolve quickly, but some take longer.  
        **P90 Resolution Time**: {p90_resolution:.0f} minutes (90% of tickets resolve within this time)  
        **Action**: Investigate tickets beyond P90 - they may require process improvements.
        """)
    
    with col2:
        st.subheader("⚡ First Response Time Distribution")
        fig = px.histogram(filtered_tickets, x='first_response_time_min', nbins=50, title='First Response Time (minutes)')
        st.plotly_chart(fig, use_container_width=True)
        p50_response = np.percentile(filtered_tickets['first_response_time_min'].dropna(), 50)
        st.markdown(f"""
        **Analysis**: Fast first response correlates strongly with higher CSAT.  
        **Median Response Time**: {p50_response:.0f} minutes  
        **Action**: Aim for median response <5 minutes during business hours.
        """)
    
    # Ticket table
    st.subheader("📋 Recent Tickets")
    st.dataframe(filtered_tickets[['ticket_id', 'ticket_created_at', 'issue_type', 'priority', 'status', 'csat_score']].head(50))

# Agent Performance
elif page == "Agent Performance":
    st.title("👥 Agent Performance")
    st.markdown("""
    Evaluate individual agent performance against key metrics. Identify top performers and areas for improvement.
    """)
    
    agent_data = data['agent_performance']
    
    # Top performers by CSAT
    st.subheader("🏆 Top Agents by CSAT")
    top_csat = agent_data.sort_values('avg_csat', ascending=False).head(10)
    fig = px.bar(top_csat, x='agent_name', y='avg_csat', title='Top 10 Agents by CSAT', color='avg_csat', color_continuous_scale='Greens')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Analysis**: Recognize top performers. Their techniques can be shared with the team.  
    **Formula**: Avg CSAT = (Sum of CSAT scores for agent) / (Number of rated interactions)  
    **Best Practice**: Pair high-performing agents with newer team members for mentoring.
    """)
    
    # Top performers by tickets handled
    st.subheader("💪 Top Agents by Tickets Handled")
    top_tickets = agent_data.sort_values('tickets_handled', ascending=False).head(10)
    fig = px.bar(top_tickets, x='agent_name', y='tickets_handled', title='Top 10 Agents by Tickets Handled', color='tickets_handled', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Analysis**: High volume can indicate efficiency but may also signal burnout risk.  
    **Formula**: Tickets handled = Count of tickets closed by agent  
    **Caution**: High volume with low CSAT may indicate rushed service. Balance is key.
    """)
    
    # Performance metrics
    st.subheader("📊 Agent Performance Metrics")
    st.markdown("""
    **Key Metrics Explained**:
    - **Avg First Response**: Time to first reply (target: <5 min)
    - **Avg Resolution**: Time from creation to closure
    - **Solved Rate**: % of tickets resolved vs assigned
    - **Avg CSAT**: Customer satisfaction score
    """)
    
    # Add performance tiers
    agent_data['performance_tier'] = pd.cut(
        agent_data['avg_csat'],
        bins=[0, 3.5, 4.2, 5],
        labels=['Needs Improvement', 'Solid Performer', 'Top Performer']
    )
    
    st.dataframe(agent_data[['agent_name', 'tickets_handled', 'avg_first_response_min', 'avg_resolution_min', 'avg_csat', 'solved_rate', 'performance_tier']].sort_values('avg_csat', ascending=False))
    
    # Performance insights
    avg_team_csat = agent_data['avg_csat'].mean()
    avg_team_resolution = agent_data['avg_resolution_min'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Team Average CSAT", f"{avg_team_csat:.2f}")
    col2.metric("Team Avg Resolution Time", f"{avg_team_resolution:.0f} min")
    
    st.markdown("""
    **Recommendations**:
    1. **For low performers**: Schedule one-on-one coaching sessions
    2. **For high performers**: Consider promotion or lead roles
    3. **Balance workload**: Ensure even distribution of ticket volume
    4. **Cross-training**: Rotate agents across issue types for skill development
    """)

# Issue Analytics
elif page == "Issue Analytics":
    st.title("🔍 Issue Analytics")
    st.markdown("""
    Analyze ticket patterns by issue type to identify systemic problems and prioritize improvements.
    """)
    
    issue_data = data['issue_type_metrics']
    
    # Issue comparison
    st.subheader("📊 Issue Type Comparison")
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Total Tickets', 'Avg Resolution Time', 'Avg CSAT', 'SLA Breach Rate'))
    
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['total_tickets']), row=1, col=1)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['avg_resolution_min']), row=1, col=2)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['avg_csat']), row=2, col=1)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['sla_breach_rate']), row=2, col=2)
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Analysis Framework**:
    | Issue Type | High Volume | Low CSAT | Long Resolution | High SLA Breach | Action |
    |------------|-------------|----------|-----------------|-----------------|--------|
    | **Yes** | **Yes** | **Yes** | **Yes** | Critical - Immediate investigation |
    | **Yes** | No | No | No | High volume - Optimize processes |
    | No | **Yes** | **Yes** | No | Quality issue - Training needed |
    | No | No | **Yes** | **Yes** | Complex issue - Escalation path review |
    """)
    
    # Issue details with insights
    st.subheader("📋 Issue Type Details")
    
    # Add calculated metrics
    issue_data['volume_rank'] = issue_data['total_tickets'].rank(ascending=False).astype(int)
    issue_data['csat_rank'] = issue_data['avg_csat'].rank(ascending=True).astype(int)
    issue_data['priority_score'] = (issue_data['volume_rank'] * 0.4 + issue_data['csat_rank'] * 0.6).round(2)
    
    st.dataframe(issue_data[['issue_type', 'total_tickets', 'avg_resolution_min', 'avg_csat', 'sla_breach_rate', 'volume_rank', 'csat_rank', 'priority_score']].sort_values('priority_score'))
    
    st.markdown("""
    **Priority Score Formula**: (Volume Rank × 0.4) + (CSAT Rank × 0.6)  
    Lower score = Higher priority for improvement
    
    **Action Recommendations**:
    - **High volume, low CSAT**: Invest in self-service knowledge base articles
    - **Long resolution times**: Review escalation paths and available resources
    - **High SLA breaches**: Adjust SLAs or increase staffing for that issue type
    - **Low volume but critical**: Create specialized handling procedures
    """)

# SLA Monitoring
elif page == "SLA Monitoring":
    st.title("⏱️ SLA Monitoring")
    st.markdown("""
    Track Service Level Agreement (SLA) compliance. Identify breaches and optimize response times.
    """)
    
    daily_sla = data['daily_metrics'].sort_values('created_date')
    priority_sla = data['priority_metrics']
    tickets = data['tickets']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily SLA Breach Rate")
        fig = px.line(daily_sla, x='created_date', y='sla_breach_rate', title='SLA Breach Rate Trend')
        st.plotly_chart(fig, use_container_width=True)
        breach_trend = daily_sla['sla_breach_rate'].pct_change().mean() * 100
        st.markdown(f"""
        **Analysis**: Track breach rate over time. Target should be <5%.  
        **Trend**: {breach_trend:+.1f}% average daily change  
        **Action**: Investigate spikes - correlate with ticket volume, staffing, or incidents.
        """)
    
    with col2:
        st.subheader("🚨 SLA Breach by Priority")
        fig = px.bar(priority_sla, x='priority', y='sla_breach_rate', color='priority', title='SLA Breach Rate by Priority')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: Critical tickets should have lowest breach rates.  
        **Expectation**: Critical < High < Medium < Low  
        **Action**: If critical breach rate is high, review escalation procedures immediately.
        """)
    
    # SLA breach statistics
    total_breaches = tickets['sla_breach'].sum()
    breach_rate = (total_breaches / len(tickets)) * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Total SLA Breaches", total_breaches)
    col2.metric("Overall Breach Rate", f"{breach_rate:.1f}%")
    
    # SLA breach details
    st.subheader("📋 Recent SLA Breach Tickets")
    sla_breach_tickets = tickets[tickets['sla_breach'] == True].head(50)
    st.dataframe(sla_breach_tickets[['ticket_id', 'priority', 'issue_type', 'resolution_time_min', 'agent_name']])
    
    st.markdown("""
    **SLA Improvement Strategies**:
    1. **Monitor in real-time**: Set alerts for breach rate >5%
    2. **Prioritize critical**: Ensure critical tickets get immediate attention
    3. **Capacity planning**: Schedule agents based on historical volume patterns
    4. **Automate routing**: Use ML models to route tickets to best-suited agents
    5. **Review SLAs**: Ensure SLAs are realistic for current capacity
    
    **Breach Root Cause Analysis**:
    - Check if breaches cluster by agent, time period, or issue type
    - Investigate tickets with longest resolution times
    - Survey customers who experienced breaches
    """)

# CSAT Insights
elif page == "CSAT Insights":
    st.title("😊 CSAT Insights")
    st.markdown("""
    Deep dive into Customer Satisfaction (CSAT) metrics. Understand drivers of satisfaction and identify improvement opportunities.
    """)
    
    daily_csat = data['daily_metrics'].sort_values('created_date')
    issue_csat = data['issue_type_metrics']
    priority_csat = data['priority_metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 CSAT Trend")
        fig = px.line(daily_csat, x='created_date', y='avg_csat', title='Daily CSAT Score')
        st.plotly_chart(fig, use_container_width=True)
        csat_trend = daily_csat['avg_csat'].pct_change().mean() * 100
        st.markdown(f"""
        **Analysis**: Track overall satisfaction trend.  
        **Trend**: {csat_trend:+.2f}% average daily change  
        **Target**: Maintain above 4.0, aim for 4.5+
        """)
    
    with col2:
        st.subheader("⚠️ Low CSAT Rate Trend")
        fig = px.line(daily_csat, x='created_date', y='low_csat_rate', title='Daily Low CSAT Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Definition**: Low CSAT = Score ≤ 2 (out of 5)  
        **Target**: Keep below 10%  
        **Action**: Investigate days with spikes - correlate with incidents or staffing changes.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 CSAT by Issue Type")
        fig = px.bar(issue_csat, x='issue_type', y='avg_csat', title='CSAT by Issue Type')
        st.plotly_chart(fig, use_container_width=True)
        lowest_csat_issue = issue_csat.loc[issue_csat['avg_csat'].idxmin()]
        st.markdown(f"""
        **Worst Performing**: {lowest_csat_issue['issue_type']} with CSAT {lowest_csat_issue['avg_csat']:.2f}  
        **Action**: Focus on improving support for this issue type - add training, create better documentation, or simplify resolution process.
        """)
    
    with col2:
        st.subheader("🚨 CSAT by Priority")
        fig = px.bar(priority_csat, x='priority', y='avg_csat', color='priority', title='CSAT by Priority')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Analysis**: Higher priority tickets often have lower CSAT due to urgency and customer stress.  
        **Expectation**: Critical tickets may have slightly lower CSAT but should still meet minimum standards.  
        **Action**: Ensure critical tickets receive extra attention and empathy training.
        """)
    
    # CSAT Drivers Analysis
    st.subheader("🎯 CSAT Drivers Analysis")
    st.markdown("""
    **Key Drivers of CSAT**:
    | Factor | Impact | How to Improve |
    |--------|--------|----------------|
    | **First Response Time** | High | Reduce initial wait; set expectations |
    | **Resolution Time** | High | Streamline processes; empower agents |
    | **Agent Friendliness** | Medium | Soft skills training; empathy programs |
    | **Communication** | Medium | Proactive updates; clear explanations |
    | **Issue Resolution** | Critical | Fix root causes; prevent recurrence |
    
    **CSAT Improvement Formula**:
    ```
    CSAT Improvement = (Training × 0.3) + (Process Optimization × 0.4) + (Technology × 0.3)
    ```
    
    **Quick Wins**:
    1. Implement automated acknowledgements for new tickets
    2. Provide agents with quick-reference knowledge base
    3. Add "happy path" workflows for common issues
    4. Schedule follow-ups for low CSAT interactions
    """)

# Trend Analysis
elif page == "Trend Analysis":
    st.title("📈 Trend Analysis")
    st.markdown("""
    Identify patterns and predict future performance. Use historical data to inform strategic decisions.
    """)
    
    daily_data = data['daily_metrics'].sort_values('created_date')
    tickets = data['tickets']
    
    # Multi-trend chart
    st.subheader("🔄 Key Metrics Correlation")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=daily_data['created_date'], y=daily_data['total_tickets'], name='Total Tickets'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_data['created_date'], y=daily_data['avg_csat'], name='Avg CSAT'), secondary_y=True)
    fig.add_trace(go.Scatter(x=daily_data['created_date'], y=daily_data['solved_rate'], name='Solved Rate'), secondary_y=True)
    
    fig.update_layout(height=500)
    fig.update_yaxes(title_text="Total Tickets", secondary_y=False)
    fig.update_yaxes(title_text="CSAT / Rate", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlations
    ticket_csat_corr = daily_data['total_tickets'].corr(daily_data['avg_csat'])
    ticket_solved_corr = daily_data['total_tickets'].corr(daily_data['solved_rate'])
    
    st.markdown(f"""
    **Correlation Analysis**:
    - **Ticket Volume vs CSAT**: {ticket_csat_corr:.2f} (negative = higher volume → lower CSAT)
    - **Ticket Volume vs Solved Rate**: {ticket_solved_corr:.2f}
    
    **Interpretation**:
    - Negative correlation between volume and CSAT is expected (more tickets = less time per customer)
    - If correlation is strongly negative (<-0.5), consider capacity constraints
    
    **Action**: Use these trends to predict staffing needs. When volume increases by X%, CSAT typically decreases by Y%.
    """)
    
    # Day of week analysis
    st.subheader("📅 Day of Week Analysis")
    dow_data = tickets.groupby('created_dayofweek').agg(
        total_tickets=('ticket_id', 'count'),
        avg_csat=('csat_score', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean'),
        avg_first_response_min=('first_response_time_min', 'mean')
    ).reset_index()
    dow_data['avg_csat'] = dow_data['avg_csat'].round(2)
    dow_data['avg_resolution_min'] = dow_data['avg_resolution_min'].round(0)
    dow_data['avg_first_response_min'] = dow_data['avg_first_response_min'].round(0)
    dow_data['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig = make_subplots(rows=1, cols=4, subplot_titles=('Tickets by Day', 'CSAT by Day', 'Resolution Time by Day', 'First Response by Day'))
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['total_tickets']), row=1, col=1)
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['avg_csat']), row=1, col=2)
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['avg_resolution_min']), row=1, col=3)
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['avg_first_response_min']), row=1, col=4)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak day analysis
    peak_day = dow_data.loc[dow_data['total_tickets'].idxmax()]
    lowest_csat_day = dow_data.loc[dow_data['avg_csat'].idxmin()]
    
    st.markdown(f"""
    **Key Findings**:
    - **Peak Day**: {peak_day['day_name']} with {peak_day['total_tickets']} tickets
    - **Lowest CSAT**: {lowest_csat_day['day_name']} with CSAT {lowest_csat_day['avg_csat']}
    
    **Workforce Planning Recommendations**:
    1. **Staff accordingly**: Schedule extra agents for peak days (typically Monday)
    2. **Weekend coverage**: If weekends show long response times, consider on-call rotations
    3. **Friday preparation**: Ensure knowledge base is updated before weekend
    4. **Monday strategy**: Implement priority queues to handle backlog efficiently
    """)
    
    # Hour of day analysis
    st.subheader("🕐 Hour of Day Analysis")
    hod_data = tickets.groupby('created_hour').agg(
        total_tickets=('ticket_id', 'count'),
        avg_csat=('csat_score', 'mean')
    ).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=hod_data['created_hour'], y=hod_data['total_tickets'], name='Tickets'), secondary_y=False)
    fig.add_trace(go.Line(x=hod_data['created_hour'], y=hod_data['avg_csat'], name='CSAT'), secondary_y=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Analysis**:
    - Identify peak hours (typically 10 AM - 2 PM for B2B)
    - Note CSAT dips during peak times
    - Schedule breaks and shift handoffs outside peak hours
    
    **Strategic Recommendations**:
    1. **Predictive scaling**: Use historical patterns to adjust staffing levels
    2. **Automation**: Deploy chatbots for common issues during peak hours
    3. **Self-service promotion**: Highlight knowledge base during busy periods
    4. **Real-time monitoring**: Set alerts for unusual spikes
    """)

# Experiment Analysis
elif page == "Experiment Analysis":
    st.title("🧪 Experiment Analysis")
    st.markdown("""
    ### A/B Testing and Statistical Analysis Framework
    
    Design, run, and analyze experiments to validate the value and business impact 
    of product changes and tooling enhancements. This module provides comprehensive 
    statistical testing capabilities following industry best practices.
    
    **Use Cases**:
    - Test new agent tooling features
    - Compare support workflow efficiency
    - Validate CSAT improvement initiatives
    - Measure SLA enhancement impact
    """)
    
    # Introduction section
    st.markdown("""
    ---
    ### 📚 Statistical Methodology
    
    This analysis framework implements industry-standard statistical methods:
    
    | Method | Purpose | Application |
    |--------|---------|------------|
    | **t-test (Independent)** | Compare means between two groups | Control vs Variant CSAT |
    | **t-test (Paired)** | Compare means before/after change | Pre vs Post resolution time |
    | **Chi-Square Test** | Compare proportions/rates | SLA breach rates, conversion rates |
    | **Mann-Whitney U** | Non-parametric group comparison | When data is not normally distributed |
    | **ANOVA** | Compare means across multiple groups | CSAT by channel, resolution by priority |
    | **Correlation Analysis** | Measure relationship strength | Volume vs CSAT, response time vs CSAT |
    
    **Statistical Significance**:
    - α (alpha) = 0.05 (95% confidence level)
    - p-value < 0.05 indicates statistical significance
    - Effect size (Cohen's d) measures practical significance
    """)
    
    # Load or generate experiment data
    @st.cache_data
    def load_ab_test_data():
        import os
        ab_data_path = 'data/processed/ab_test_data.csv'
        if os.path.exists(ab_data_path):
            return pd.read_csv(ab_data_path)
        return None
    
    ab_data = load_ab_test_data()
    
    # Sample size calculator
    st.markdown("""
    ---
    ### 📐 Sample Size Calculator
    
    Before running an experiment, calculate the required sample size to detect 
    your minimum desired effect with sufficient statistical power.
    """)
    
    col1, col2, col3 = st.columns(3)
    baseline_rate = col1.number_input("Baseline Rate (%)", value=10.0, min_value=0.1, max_value=100.0) / 100
    min_detectable_effect = col2.number_input("Min Effect to Detect (%)", value=5.0, min_value=1.0, max_value=50.0) / 100
    alpha = col3.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    
    power = 0.80
    
    # Calculate sample size (simplified formula)
    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)
    z_alpha = 1.96 if alpha == 0.05 else (2.58 if alpha == 0.01 else 1.645)
    z_beta = 0.84
    
    pooled_p = (p1 + p2) / 2
    n_per_group = ((z_alpha * np.sqrt(2 * pooled_p * (1 - pooled_p)) + 
                   z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / 
                  (p2 - p1)) ** 2
    
    total_samples = int(np.ceil(2 * n_per_group))
    
    col1, col2 = st.columns(2)
    col1.metric("Samples Per Group", f"{int(np.ceil(n_per_group)):,}")
    col2.metric("Total Samples Needed", f"{total_samples:,}")
    
    st.markdown(f"""
    **Formula**:
    ```
    n = (z_α × √(2p̄(1-p̄)) + z_β × √(p₁(1-p₁) + p₂(1-p₂)))² / (p₂ - p₁)²
    
    Where:
    - z_α = {z_alpha:.3f} (for α = {alpha})
    - z_β = {z_beta:.2f} (for 80% power)
    - p̄ = {pooled_p:.4f} (pooled proportion)
    ```
    
    **Duration Estimate**: 
    Assuming ~{max(1, total_samples // 200)} tickets/day per group → **{max(1, (total_samples // 200) * 2)} days** to reach sample size
    """)
    
    # A/B Test Results Analysis
    st.markdown("""
    ---
    ### 📊 A/B Test Results Analysis
    
    Analyze experiment results using statistical hypothesis testing.
    """)
    
    # Use cached data or show demo
    if ab_data is not None and 'experiment_group' in ab_data.columns:
        control = ab_data[ab_data['experiment_group'] == 'control']
        variant = ab_data[ab_data['experiment_group'] == 'variant']
        
        # Tabs for different metrics
        test_tabs = st.tabs(["CSAT Analysis", "Resolution Time", "SLA Breach Rate"])
        
        with test_tabs[0]:
            st.subheader("Customer Satisfaction (CSAT) Score Comparison")
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            control_csat_mean = control['csat_score'].mean()
            variant_csat_mean = variant['csat_score'].mean()
            csat_lift = (variant_csat_mean - control_csat_mean) / control_csat_mean * 100
            
            col1.metric("Control CSAT", f"{control_csat_mean:.3f}")
            col2.metric("Variant CSAT", f"{variant_csat_mean:.3f}")
            col3.metric("Absolute Lift", f"{variant_csat_mean - control_csat_mean:+.3f}")
            col4.metric("Relative Lift", f"{csat_lift:+.1f}%", 
                       delta="Significant" if abs(csat_lift) > 1 else "Not Significant")
            
            # Distribution plot
            fig = make_subplots(rows=1, cols=2, subplot_titles=('CSAT Distribution', 'Box Plot Comparison'))
            
            fig.add_trace(go.Histogram(x=control['csat_score'], name='Control', opacity=0.7, 
                                      marker_color='#1f77b4'), row=1, col=1)
            fig.add_trace(go.Histogram(x=variant['csat_score'], name='Variant', opacity=0.7,
                                      marker_color='#2ca02c'), row=1, col=1)
            
            fig.add_trace(go.Box(y=control['csat_score'], name='Control', 
                                marker_color='#1f77b4'), row=1, col=2)
            fig.add_trace(go.Box(y=variant['csat_score'], name='Variant',
                                marker_color='#2ca02c'), row=1, col=2)
            
            fig.update_layout(height=400, barmode='overlay', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical test results
            from scipy.stats import ttest_ind, mannwhitneyu
            
            # T-test
            t_stat, t_pvalue = ttest_ind(control['csat_score'], variant['csat_score'], equal_var=False)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((control['csat_score'].std()**2 + variant['csat_score'].std()**2) / 2)
            cohens_d = (variant_csat_mean - control_csat_mean) / pooled_std
            
            st.markdown(f"""
            #### Statistical Test Results
            
            **Independent Samples t-test (Welch's)**
            
            | Metric | Value |
            |--------|-------|
            | Control Mean | {control_csat_mean:.4f} |
            | Variant Mean | {variant_csat_mean:.4f} |
            | Mean Difference | {variant_csat_mean - control_csat_mean:.4f} |
            | t-statistic | {t_stat:.4f} |
            | p-value | {t_pvalue:.6f} |
            | Cohen's d | {cohens_d:.4f} |
            | Effect Size | {'Negligible' if abs(cohens_d) < 0.2 else 'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'} |
            | **Conclusion** | **{'Statistically Significant' if t_pvalue < 0.05 else 'Not Statistically Significant'} at α=0.05** |
            
            **Interpretation Guide**:
            - **p-value < 0.05**: Results are unlikely due to chance (95% confidence)
            - **Cohen's d**: 0.2=small, 0.5=medium, 0.8=large practical effect
            - **Business Impact**: Consider both statistical and practical significance
            """)
        
        with test_tabs[1]:
            st.subheader("Resolution Time Comparison")
            
            control_res = control['resolution_time_min'].mean()
            variant_res = variant['resolution_time_min'].mean()
            res_lift = (control_res - variant_res) / control_res * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Control Avg", f"{control_res:.1f} min")
            col2.metric("Variant Avg", f"{variant_res:.1f} min")
            col3.metric("Time Saved", f"{control_res - variant_res:.1f} min")
            col4.metric("Improvement", f"{res_lift:.1f}%", 
                       delta="Significant" if res_lift > 5 else "Not Significant")
            
            # Histogram
            fig = px.histogram(pd.concat([control.assign(group='Control'), 
                                         variant.assign(group='Variant')]),
                             x='resolution_time_min', color='group', barmode='overlay',
                             opacity=0.7, nbins=50)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = mannwhitneyu(control['resolution_time_min'], 
                                          variant['resolution_time_min'], alternative='two-sided')
            
            st.markdown(f"""
            #### Mann-Whitney U Test (Non-parametric)
            
            Non-parametric test does not assume normal distribution.
            
            | Metric | Value |
            |--------|-------|
            | U-statistic | {u_stat:,.0f} |
            | p-value | {u_pvalue:.6f} |
            | **Conclusion** | **{'Statistically Significant' if u_pvalue < 0.05 else 'Not Statistically Significant'}** |
            """)
        
        with test_tabs[2]:
            st.subheader("SLA Breach Rate Analysis")
            
            control_breach = control['sla_breach'].mean() * 100
            variant_breach = variant['sla_breach'].mean() * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Control Breach Rate", f"{control_breach:.2f}%")
            col2.metric("Variant Breach Rate", f"{variant_breach:.2f}%")
            col3.metric("Absolute Change", f"{variant_breach - control_breach:+.2f}%")
            col4.metric("Relative Change", f"{(variant_breach - control_breach) / control_breach * 100 if control_breach > 0 else 0:+.1f}%",
                       delta="Significant" if abs(variant_breach - control_breach) > 1 else "Not Significant")
            
            # Chi-square test for proportions
            from scipy.stats import chi2_contingency
            
            contingency = np.array([
                [len(control) - control['sla_breach'].sum(), control['sla_breach'].sum()],
                [len(variant) - variant['sla_breach'].sum(), variant['sla_breach'].sum()]
            ])
            
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V
            n = len(control) + len(variant)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            st.markdown(f"""
            #### Chi-Square Test of Independence
            
            | Metric | Value |
            |--------|-------|
            | χ² statistic | {chi2:.4f} |
            | Degrees of Freedom | {dof} |
            | p-value | {p_value:.6f} |
            | Cramér's V | {cramers_v:.4f} |
            | **Conclusion** | **{'Statistically Significant' if p_value < 0.05 else 'Not Statistically Significant'}** |
            """)
    
    else:
        st.info("📌 No experiment data found. Run `scripts/ab_test_analysis.py` to generate demo data, or upload your experiment results.")
        
        # Demo visualization
        st.markdown("""
        ---
        ### 📈 Sample Experiment Dashboard Preview
        
        Below is a preview of the type of analysis you can perform with experiment data:
        """)
        
        # Generate demo data
        np.random.seed(42)
        demo_days = 30
        demo_dates = pd.date_range(start='2024-02-01', periods=demo_days, freq='D')
        
        demo_data = pd.DataFrame({
            'date': demo_dates,
            'control_csat': np.random.normal(4.1, 0.2, demo_days),
            'variant_csat': np.random.normal(4.25, 0.18, demo_days),
            'control_resolution': np.random.normal(220, 30, demo_days),
            'variant_resolution': np.random.normal(200, 28, demo_days)
        })
        
        # Trend comparison
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=demo_data['date'], y=demo_data['control_csat'], 
                                name='Control CSAT', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=demo_data['date'], y=demo_data['variant_csat'],
                                name='Variant CSAT', line=dict(color='#2ca02c')))
        
        fig.update_layout(title='Daily CSAT Comparison (Demo Data)', height=400)
        fig.update_yaxes(title_text="CSAT Score")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **To get started with real experiment analysis:**
        1. Run the A/B test analysis script: `python scripts/ab_test_analysis.py`
        2. Upload your experiment data in CSV format
        3. Select metrics to analyze
        4. View statistical test results and recommendations
        """)
    
    # Best Practices
    st.markdown("""
    ---
    ### ✅ A/B Testing Best Practices
    
    **Design Phase**:
    1. Define clear success metrics (primary + secondary)
    2. Calculate required sample size before starting
    3. Set run time based on sample size, not calendar convenience
    4. Document hypothesis and expected impact
    
    **Analysis Phase**:
    1. Check for sample ratio mismatch (SRM) - indicates data issues
    2. Look at both statistical significance AND practical significance
    3. Consider segment analysis (new vs existing users, by channel, etc.)
    4. Account for multiple comparisons if testing many metrics
    
    **Reporting Phase**:
    1. Present confidence intervals, not just p-values
    2. Include effect size and business impact
    3. Document learnings, even for inconclusive tests
    4. Create follow-up experiments based on findings
    
    **Common Pitfalls to Avoid**:
    - 🚫 Peeking at results before reaching sample size
    - 🚫 Stopping test early based on significance alone
    - 🚫 Ignoring segment-level effects
    - 🚫 Testing too many variants without correction
    - 🚫 Drawing conclusions from underpowered tests
    """)

# Time Series Forecasting
elif page == "Time Series Forecasting":
    st.title("📈 Time Series Forecasting")
    st.markdown("""
    ### Predictive Analytics for Support Metrics
    
    Forecast future support metrics to enable proactive decision-making,
    capacity planning, and resource allocation. This module provides
    comprehensive time series analysis capabilities.
    
    **Use Cases**:
    - Predict ticket volume for staffing planning
    - Forecast CSAT trends for target setting
    - Anticipate SLA breaches before they occur
    - Identify seasonal patterns for resource optimization
    """)
    
    # Import the forecasting module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    
    try:
        from scripts.time_series_forecasting import TimeSeriesForecaster, forecast_metrics
        forecaster_available = True
    except ImportError as e:
        st.warning(f"Forecasting module not available: {e}")
        forecaster_available = False
    
    if forecaster_available:
        # Select metric to forecast
        st.markdown("""
        ---
        ### Select Metrics for Forecasting
        """)
        
        metric_options = ['total_tickets', 'avg_csat', 'solved_rate', 'sla_breach_rate']
        selected_metric = st.selectbox("Select Metric", metric_options)
        
        # Get data
        daily_data = data['daily_metrics'].copy()
        daily_data['created_date'] = pd.to_datetime(daily_data['created_date'])
        daily_data = daily_data.sort_values('created_date')
        
        # Forecasting parameters
        col1, col2, col3 = st.columns(3)
        forecast_periods = col1.number_input("Forecast Periods (days)", value=7, min_value=1, max_value=30)
        model_type = col2.selectbox("Model Type", ["auto", "arima", "ets"])
        
        if st.button("Generate Forecast"):
            with st.spinner("Running forecast analysis..."):
                # Run complete forecast analysis
                forecast_result = forecast_metrics(
                    daily_data,
                    'created_date',
                    selected_metric,
                    forecast_periods=forecast_periods,
                    model=model_type
                )
                
                # Display results
                st.markdown("""
                ---
                ### Forecast Results
                """)
                
                # Trend analysis
                col1, col2 = st.columns(2)
                trend = forecast_result['trend']
                col1.metric("Trend Direction", trend['trend_type'])
                col1.markdown(f"""
                **Trend Strength**: {trend['trend_strength']}  
                **Daily Change**: {trend['daily_change']:.4f}  
                **R-squared**: {trend['r_squared']:.4f}
                """)
                
                # Seasonality
                seasonality = forecast_result['seasonality']
                col2.metric("Seasonality", "Detected" if seasonality['has_seasonality'] else "Not Detected")
                if seasonality['has_seasonality']:
                    col2.markdown(f"""
                    **Period**: {seasonality['period']} days  
                    **Strength**: {seasonality['seasonal_strength']:.2%}  
                    **Peak Factor**: {seasonality['peak_seasonal_factor']:.4f}
                    """)
                
                # Forecast values
                st.markdown("""
                ---
                ### Predicted Values
                """)
                
                forecast = forecast_result['forecast']
                forecast_df = pd.DataFrame({
                    'Date': forecast['forecast_dates'],
                    'Predicted': forecast['forecast'],
                    'Lower CI (95%)': forecast['ci_lower'],
                    'Upper CI (95%)': forecast['ci_upper']
                })
                
                st.dataframe(forecast_df.style.format({
                    'Predicted': '{:.2f}',
                    'Lower CI (95%)': '{:.2f}',
                    'Upper CI (95%)': '{:.2f}'
                }))
                
                # Forecast chart
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=daily_data['created_date'].tail(30),
                    y=daily_data[selected_metric].tail(30),
                    name='Historical',
                    mode='lines',
                    line=dict(color='#1f77b4')
                ))
                
                # Forecast
                forecast_dates = [pd.to_datetime(d) for d in forecast['forecast_dates']]
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast['forecast'],
                    name='Forecast',
                    mode='lines',
                    line=dict(color='#ff7f0e', dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=forecast['ci_upper'] + forecast['ci_lower'][::-1],
                    name='95% CI',
                    fill='toself',
                    opacity=0.2,
                    line=dict(color='#ff7f0e', width=0)
                ))
                
                fig.update_layout(
                    title=f'{selected_metric} - Historical and Forecast',
                    xaxis_title='Date',
                    yaxis_title=selected_metric,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model interpretation
                st.markdown(f"""
                **Interpretation**: {forecast['interpretation']}  
                **Model Used**: {forecast_result['model_used']}
                """)
                
                # Anomaly detection
                st.markdown("""
                ---
                ### Anomaly Detection
                """)
                
                anomalies = forecast_result['anomalies']
                col1, col2 = st.columns(2)
                col1.metric("Anomalies Detected", anomalies['anomaly_count'])
                col2.metric("Anomaly Rate", f"{anomalies['anomaly_rate']:.2f}%")
                
                if anomalies['anomaly_count'] > 0:
                    st.markdown("**Anomaly Dates**:")
                    for date, value in zip(anomalies['anomaly_dates'][:5], anomalies['anomaly_values'][:5]):
                        st.markdown(f"- {date}: {value:.4f}")
                
                # Accuracy metrics
                st.markdown("""
                ---
                ### Model Information
                """)
                
                if 'aic' in forecast and forecast['aic']:
                    st.markdown(f"**AIC**: {forecast['aic']:.2f}")
                
                st.markdown(f"""
                **Statistical Methods Used**:
                - Trend detection via linear regression
                - Seasonality via classical decomposition
                - ARIMA/ETS for forecasting
                - IQR-based anomaly detection
                - CUSUM for change point detection
                """)
    else:
        st.info("Forecasting module not loaded. Please ensure scripts/time_series_forecasting.py exists.")

# CATE Analysis
elif page == "CATE Analysis":
    st.title("🎯 CATE Analysis (Heterogeneous Treatment Effects)")
    st.markdown("""
    ### Causal AI for Targeted Decision-Making

    Estimate Conditional Average Treatment Effects (CATE) to understand
    how treatment effects vary across different subgroups. This enables
    targeted interventions and personalized resource allocation.

    **Use Cases**:
    - Identify which customer segments benefit most from proactive support
    - Determine which agents respond best to specific training programs
    - Optimize tooling investments for different team types
    - Personalize escalation strategies based on customer characteristics
    """)

    # Import CATE module
    try:
        from scripts.causal_inference import CausalInferenceAnalyzer
        from scripts.cate_meta_learners import MetaLearnerCATE
        cate_available = True
    except ImportError as e:
        st.warning(f"CATE module not available: {e}")
        cate_available = False

    if cate_available:
        # Generate demo data for illustration
        st.markdown("""
        ---
        ### CATE Analysis Demo

        The following demonstrates CATE analysis on support ticket data to identify
        heterogeneous treatment effects across different customer segments.
        """)

        # Create demo data with known heterogeneity
        np.random.seed(42)
        n_demo = 500

        demo_df = pd.DataFrame({
            'customer_tier': np.random.choice(['Standard', 'Premium', 'Enterprise'], n_demo),
            'ticket_complexity': np.random.uniform(1, 10, n_demo),
            'agent_experience_years': np.random.uniform(0.5, 10, n_demo),
            'response_time_historical': np.random.uniform(10, 120, n_demo),
            'treatment': np.random.binomial(1, 0.5, n_demo),
            'csat_score': np.zeros(n_demo)
        })

        # Heterogeneous treatment effect: premium customers respond better to proactive support
        treatment_effect = (
            0.5 +
            0.3 * (demo_df['customer_tier'] == 'Premium').astype(int) +
            0.2 * (demo_df['customer_tier'] == 'Enterprise').astype(int) +
            0.1 * demo_df['ticket_complexity']
        )

        demo_df['csat_score'] = (
            3.5 +
            treatment_effect * demo_df['treatment'] +
            0.1 * demo_df['ticket_complexity'] +
            0.05 * demo_df['agent_experience_years'] +
            np.random.normal(0, 0.2, n_demo)
        )

        demo_df['csat_score'] = demo_df['csat_score'].clip(1, 5)

        # Analysis controls
        st.markdown("""
        ---
        #### Analysis Configuration
        """)

        learner_options = {
            's_learner': 'S-Learner (Single Model)',
            't_learner': 'T-Learner (Two Models)',
            'x_learner': 'X-Learner (Cross-Fitting) - Recommended',
            'dr_learner': 'DR-Learner (Doubly Robust)'
        }

        selected_learner = st.selectbox(
            "Select Meta-Learner",
            options=list(learner_options.keys()),
            format_func=lambda x: learner_options[x],
            index=2  # X-Learner by default
        )

        feature_cols = ['customer_tier', 'ticket_complexity', 'agent_experience_years', 'response_time_historical']
        selected_features = st.multiselect(
            "Select Features for Heterogeneity Analysis",
            options=feature_cols,
            default=feature_cols
        )

        if st.button("Run CATE Analysis"):
            with st.spinner("Running CATE analysis..."):
                analyzer = MetaLearnerCATE()

                if selected_learner == 's_learner':
                    result = analyzer.s_learner(
                        demo_df, 'treatment', 'csat_score', selected_features
                    )
                elif selected_learner == 't_learner':
                    result = analyzer.t_learner(
                        demo_df, 'treatment', 'csat_score', selected_features
                    )
                elif selected_learner == 'x_learner':
                    result = analyzer.x_learner(
                        demo_df, 'treatment', 'csat_score', selected_features
                    )
                else:
                    result = analyzer.dr_learner(
                        demo_df, 'treatment', 'csat_score', selected_features
                    )

                # Display results
                st.markdown("""
                ---
                #### CATE Analysis Results
                """)

                col1, col2, col3, col4 = st.columns(4)

                if 'ate' in result:
                    col1.metric("Average Treatment Effect (ATE)", f"{result['ate']:.4f}")
                    col2.metric("Standard Error", f"{result.get('ate_se', 'N/A'):.4f}" if 'ate_se' in result else "N/A")
                    col3.metric("95% CI Lower", f"{result.get('ate_ci_lower', 'N/A'):.4f}" if 'ate_ci_lower' in result else "N/A")
                    col4.metric("95% CI Upper", f"{result.get('ate_ci_upper', 'N/A'):.4f}" if 'ate_ci_upper' in result else "N/A")

                    # Treatment effect heterogeneity
                    st.markdown("""
                    ---
                    #### Treatment Effect Heterogeneity
                    """)

                    if 'heterogeneity_sources' in result:
                        het_df = []
                        for feature, het_data in result['heterogeneity_sources'].items():
                            het_df.append({
                                'Feature': feature,
                                'Correlation': het_data.get('correlation', 0),
                                'CATE Range': het_data.get('range', 0)
                            })

                        if het_df:
                            het_display = pd.DataFrame(het_df)
                            st.dataframe(het_display.style.format({
                                'Correlation': '{:.4f}',
                                'CATE Range': '{:.4f}'
                            }))

                        # Interpretation
                        st.markdown("""
                        **Interpretation Guide**:
                        - **Correlation**: Positive values indicate higher feature values → larger treatment effects
                        - **CATE Range**: Difference in treatment effects between lowest and highest feature values
                        - **Policy Implication**: Target interventions to subgroups with highest predicted treatment effects
                        """)

                    # Overall interpretation
                    st.markdown(f"""
                    ---
                    #### Summary

                    **{result.get('interpretation', 'Analysis complete')}**

                    **Key Insights**:
                    - The estimated ATE of **{result['ate']:.4f}** represents the average causal effect of the treatment
                    - Treatment effects show **{'heterogeneous' if result.get('treatment_effect_std', 0) > 0.1 else 'relatively homogeneous'}** patterns across subgroups
                    - Standard deviation of treatment effects: **{result.get('treatment_effect_std', 0):.4f}**
                    """)

                else:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")

        # Method comparison
        st.markdown("""
        ---
        #### Meta-Learner Methods Comparison
        """)

        methods_df = pd.DataFrame({
            'Method': ['S-Learner', 'T-Learner', 'X-Learner', 'DR-Learner'],
            'Strengths': [
                'Simple, good baseline',
                'Separates treatment/control modeling',
                'Handles selection bias well',
                'Doubly robust variance reduction'
            ],
            'Best For': [
                'Quick exploration',
                'When treatment effects are homogeneous',
                'General use, recommended default',
                'When both outcome and propensity models are correct'
            ],
            'Assumptions': [
                'Single model captures heterogeneity',
                'Separate models well-specified',
                'Cross-fitting reduces bias',
                'Correct specification of either outcome or propensity model'
            ]
        })

        st.dataframe(methods_df)

        st.markdown("""
        **Recommendation**: X-Learner is generally recommended as a robust default choice
        that handles heterogeneous effects well across a variety of scenarios.
        """)

    else:
        st.info("CATE module not loaded. Please ensure scripts/10_cate_meta_learners.py exists.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("SupportIQ Web - Customer Support Analytics\n\nBuilt with Streamlit • Powered by scikit-learn")