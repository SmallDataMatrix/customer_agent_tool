"""
SupportIQ Web - Streamlit Application

This is the main application for the Customer Support Analytics Dashboard.
It provides interactive visualizations and analytics for support ticket data.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

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
        'daily_metrics': pd.read_csv('data/processed/daily_metrics.csv'),
        'issue_type_metrics': pd.read_csv('data/processed/issue_type_metrics.csv'),
        'agent_performance': pd.read_csv('data/processed/agent_performance.csv'),
        'priority_metrics': pd.read_csv('data/processed/priority_metrics.csv'),
        'channel_metrics': pd.read_csv('data/processed/channel_metrics.csv'),
        'tickets': pd.read_csv('data/processed/tickets_cleaned.csv')
    }
    # Convert date columns
    if 'date' in data['daily_metrics'].columns:
        data['daily_metrics']['date'] = pd.to_datetime(data['daily_metrics']['date'])
    if 'created_date' in data['tickets'].columns:
        data['tickets']['created_date'] = pd.to_datetime(data['tickets']['created_date'])
    if 'ticket_created_at' in data['tickets'].columns:
        data['tickets']['ticket_created_at'] = pd.to_datetime(data['tickets']['ticket_created_at'])
    return data

data = load_csv_data()

# Sidebar navigation
st.sidebar.title("SupportIQ Web")
page = st.sidebar.radio("Navigation", [
    "Executive Dashboard",
    "Ticket Analysis",
    "Agent Performance",
    "Issue Analytics",
    "SLA Monitoring",
    "CSAT Insights",
    "Trend Analysis"
])

# Executive Dashboard
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown("### Key Performance Indicators")
    
    # Load metrics
    daily_metrics = data['daily_metrics'].sort_values('date', ascending=False).head(30)
    issue_metrics = data['issue_type_metrics']
    priority_metrics = data['priority_metrics']
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Tickets
    total_tickets = daily_metrics['total_tickets'].sum()
    col1.metric("Total Tickets", f"{total_tickets:,}", delta=f"{daily_metrics['total_tickets'].iloc[0] - daily_metrics['total_tickets'].iloc[1]} vs prev day")
    
    # Average CSAT
    avg_csat = daily_metrics['avg_csat'].mean()
    col2.metric("Average CSAT", f"{avg_csat:.2f}", delta=f"{avg_csat - daily_metrics['avg_csat'].iloc[1]:.2f}")
    
    # Solved Rate
    avg_solved_rate = daily_metrics['solved_rate'].mean()
    col3.metric("Solved Rate", f"{avg_solved_rate:.1f}%", delta=f"{avg_solved_rate - daily_metrics['solved_rate'].iloc[1]:.1f}%")
    
    # SLA Breach Rate
    avg_sla_breach = daily_metrics['sla_breach_rate'].mean()
    col4.metric("SLA Breach Rate", f"{avg_sla_breach:.1f}%", delta=f"{avg_sla_breach - daily_metrics['sla_breach_rate'].iloc[1]:.1f}%")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ticket Volume Trend")
        fig = px.line(daily_metrics, x='date', y='total_tickets', title='Daily Ticket Volume')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("CSAT Trend")
        fig = px.line(daily_metrics, x='date', y='avg_csat', title='Daily CSAT Score')
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tickets by Issue Type")
        fig = px.pie(issue_metrics, values='total_tickets', names='issue_type', hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tickets by Priority")
        fig = px.bar(priority_metrics, x='priority', y='total_tickets', color='priority')
        st.plotly_chart(fig, use_container_width=True)

# Ticket Analysis
elif page == "Ticket Analysis":
    st.title("🎫 Ticket Analysis")
    
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
    st.subheader("Ticket Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tickets", len(filtered_tickets))
    col2.metric("Avg Resolution Time", f"{filtered_tickets['resolution_time_min'].mean():.0f} min")
    col3.metric("Avg First Response", f"{filtered_tickets['first_response_time_min'].mean():.0f} min")
    col4.metric("Avg CSAT", f"{filtered_tickets['csat_score'].mean():.2f}")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resolution Time Distribution")
        fig = px.histogram(filtered_tickets, x='resolution_time_min', nbins=50, title='Resolution Time (minutes)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("First Response Time Distribution")
        fig = px.histogram(filtered_tickets, x='first_response_time_min', nbins=50, title='First Response Time (minutes)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Ticket table
    st.subheader("Recent Tickets")
    st.dataframe(filtered_tickets[['ticket_id', 'ticket_created_at', 'issue_type', 'priority', 'status', 'csat_score']].head(50))

# Agent Performance
elif page == "Agent Performance":
    st.title("👥 Agent Performance")
    
    agent_data = data['agent_performance']
    
    # Top performers by CSAT
    st.subheader("Top Agents by CSAT")
    top_csat = agent_data.sort_values('avg_csat', ascending=False).head(10)
    fig = px.bar(top_csat, x='agent_name', y='avg_csat', title='Top 10 Agents by CSAT')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performers by tickets handled
    st.subheader("Top Agents by Tickets Handled")
    top_tickets = agent_data.sort_values('tickets_handled', ascending=False).head(10)
    fig = px.bar(top_tickets, x='agent_name', y='tickets_handled', title='Top 10 Agents by Tickets Handled')
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.subheader("Agent Performance Metrics")
    st.dataframe(agent_data[['agent_name', 'tickets_handled', 'avg_first_response_min', 'avg_resolution_min', 'avg_csat', 'solved_rate']])

# Issue Analytics
elif page == "Issue Analytics":
    st.title("🔍 Issue Analytics")
    
    issue_data = data['issue_type_metrics']
    
    # Issue comparison
    st.subheader("Issue Type Comparison")
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Total Tickets', 'Avg Resolution Time', 'Avg CSAT', 'SLA Breach Rate'))
    
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['total_tickets']), row=1, col=1)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['avg_resolution_min']), row=1, col=2)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['avg_csat']), row=2, col=1)
    fig.add_trace(go.Bar(x=issue_data['issue_type'], y=issue_data['sla_breach_rate']), row=2, col=2)
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Issue details
    st.subheader("Issue Type Details")
    st.dataframe(issue_data)

# SLA Monitoring
elif page == "SLA Monitoring":
    st.title("⏱️ SLA Monitoring")
    
    daily_sla = data['daily_metrics'].sort_values('date')
    priority_sla = data['priority_metrics']
    tickets = data['tickets']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily SLA Breach Rate")
        fig = px.line(daily_sla, x='date', y='sla_breach_rate', title='SLA Breach Rate Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("SLA Breach by Priority")
        fig = px.bar(priority_sla, x='priority', y='sla_breach_rate', color='priority', title='SLA Breach Rate by Priority')
        st.plotly_chart(fig, use_container_width=True)
    
    # SLA breach details
    st.subheader("SLA Breach Details")
    sla_breach_tickets = tickets[tickets['sla_breach'] == True].head(100)
    st.dataframe(sla_breach_tickets[['ticket_id', 'priority', 'issue_type', 'resolution_time_min', 'agent_name']])

# CSAT Insights
elif page == "CSAT Insights":
    st.title("😊 CSAT Insights")
    
    daily_csat = data['daily_metrics'].sort_values('date')
    issue_csat = data['issue_type_metrics']
    priority_csat = data['priority_metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CSAT Trend")
        fig = px.line(daily_csat, x='date', y='avg_csat', title='Daily CSAT Score')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Low CSAT Rate Trend")
        fig = px.line(daily_csat, x='date', y='low_csat_rate', title='Daily Low CSAT Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CSAT by Issue Type")
        fig = px.bar(issue_csat, x='issue_type', y='avg_csat', title='CSAT by Issue Type')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("CSAT by Priority")
        fig = px.bar(priority_csat, x='priority', y='avg_csat', color='priority', title='CSAT by Priority')
        st.plotly_chart(fig, use_container_width=True)

# Trend Analysis
elif page == "Trend Analysis":
    st.title("📈 Trend Analysis")
    
    daily_data = data['daily_metrics'].sort_values('date')
    tickets = data['tickets']
    
    # Multi-trend chart
    st.subheader("Key Metrics Trend")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['total_tickets'], name='Total Tickets'), secondary_y=False)
    fig.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['avg_csat'], name='Avg CSAT'), secondary_y=True)
    fig.add_trace(go.Scatter(x=daily_data['date'], y=daily_data['solved_rate'], name='Solved Rate'), secondary_y=True)
    
    fig.update_layout(height=500)
    fig.update_yaxes(title_text="Total Tickets", secondary_y=False)
    fig.update_yaxes(title_text="CSAT / Rate", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.subheader("Day of Week Analysis")
    dow_data = tickets.groupby('created_dayofweek').agg(
        total_tickets=('ticket_id', 'count'),
        avg_csat=('csat_score', 'mean'),
        avg_resolution_min=('resolution_time_min', 'mean')
    ).reset_index()
    dow_data['avg_csat'] = dow_data['avg_csat'].round(2)
    dow_data['avg_resolution_min'] = dow_data['avg_resolution_min'].round(0)
    dow_data['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Tickets by Day', 'CSAT by Day', 'Resolution Time by Day'))
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['total_tickets']), row=1, col=1)
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['avg_csat']), row=1, col=2)
    fig.add_trace(go.Bar(x=dow_data['day_name'], y=dow_data['avg_resolution_min']), row=1, col=3)
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("SupportIQ Web - Customer Support Analytics")
