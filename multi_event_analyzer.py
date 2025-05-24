import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class MultiEventAnalyzer:
    def __init__(self):
        self.events = []
    
    def add_event(self, name, ticket_price, num_tickets, occupancy_rate, 
                  commission_rate, start_date, infrastructure_tier):
        """Add an event to the analysis"""
        event = {
            'name': name,
            'ticket_price': ticket_price,
            'num_tickets': num_tickets,
            'occupancy_rate': occupancy_rate,
            'commission_rate': commission_rate,
            'start_date': start_date,
            'infrastructure_tier': infrastructure_tier
        }
        self.events.append(event)
    
    def calculate_monthly_revenue_projection(self, months_ahead=12):
        """Calculate revenue projections for the next N months"""
        projection_data = []
        
        for month in range(months_ahead):
            target_date = datetime.now() + relativedelta(months=month)
            monthly_revenue = 0
            monthly_costs = 0
            events_this_month = 0
            
            for event in self.events:
                # Check if event falls in this month (simplified logic)
                if event['start_date'].month == target_date.month or month == 0:
                    tickets_sold = int(event['num_tickets'] * event['occupancy_rate'] / 100)
                    commission_revenue = tickets_sold * event['ticket_price'] * (event['commission_rate'] / 100)
                    
                    # Calculate costs (simplified)
                    infrastructure_cost = self.get_infrastructure_cost(event['infrastructure_tier'])
                    variable_costs = tickets_sold * 0.053  # Combined per-ticket costs
                    
                    monthly_revenue += commission_revenue
                    monthly_costs += infrastructure_cost + variable_costs
                    events_this_month += 1
            
            projection_data.append({
                'Month': target_date.strftime('%Y-%m'),
                'Revenue': monthly_revenue,
                'Costs': monthly_costs,
                'Net Profit': monthly_revenue - monthly_costs,
                'Events': events_this_month
            })
        
        return pd.DataFrame(projection_data)
    
    def get_infrastructure_cost(self, tier):
        """Get infrastructure cost for a tier"""
        costs = {
            'Básico': 964.00,
            'Moderado': 1445.00,
            'Intenso': 2535.00,
            'Máximo': 6166.00
        }
        return costs.get(tier, 1445.00) + 50  # Add seats.io base cost
    
    def calculate_event_roi(self):
        """Calculate ROI for each event"""
        roi_data = []
        
        for event in self.events:
            tickets_sold = int(event['num_tickets'] * event['occupancy_rate'] / 100)
            commission_revenue = tickets_sold * event['ticket_price'] * (event['commission_rate'] / 100)
            
            # Calculate total costs
            infrastructure_cost = self.get_infrastructure_cost(event['infrastructure_tier'])
            variable_costs = tickets_sold * 0.053
            total_costs = infrastructure_cost + variable_costs
            
            # Calculate ROI
            net_profit = commission_revenue - total_costs
            roi = (net_profit / total_costs * 100) if total_costs > 0 else 0
            
            roi_data.append({
                'Event': event['name'],
                'Commission Revenue': commission_revenue,
                'Total Costs': total_costs,
                'Net Profit': net_profit,
                'ROI (%)': roi,
                'Commission Rate (%)': event['commission_rate']
            })
        
        return pd.DataFrame(roi_data)

def show_multi_event_analysis():
    """Display multi-event analysis interface"""
    st.title("📅 Multi-Event Revenue Analysis")
    st.markdown("---")
    
    # Initialize session state for events
    if 'events' not in st.session_state:
        st.session_state.events = []
    
    # Event input form
    with st.expander("➕ Add New Event", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            event_name = st.text_input("Event Name", value=f"Event {len(st.session_state.events) + 1}")
            ticket_price = st.number_input("Ticket Price ($)", min_value=1.0, value=25.0, step=1.0)
            num_tickets = st.number_input("Number of Tickets", min_value=1, value=500, step=10)
            occupancy_rate = st.slider("Expected Occupancy (%)", 10, 100, 80)
        
        with col2:
            commission_rate = st.slider("Commission Rate (%)", 5, 20, 10)
            start_date = st.date_input("Event Date", value=datetime.now())
            infrastructure_tier = st.selectbox(
                "Infrastructure Tier",
                options=['Básico', 'Moderado', 'Intenso', 'Máximo'],
                index=1
            )
        
        if st.button("Add Event"):
            new_event = {
                'name': event_name,
                'ticket_price': ticket_price,
                'num_tickets': num_tickets,
                'occupancy_rate': occupancy_rate,
                'commission_rate': commission_rate,
                'start_date': start_date,
                'infrastructure_tier': infrastructure_tier
            }
            st.session_state.events.append(new_event)
            st.success(f"Added {event_name} to analysis!")
    
    # Display current events
    if st.session_state.events:
        st.subheader("📋 Current Events")
        
        # Create events dataframe for display
        events_df = pd.DataFrame(st.session_state.events)
        events_df['Tickets Sold'] = (events_df['num_tickets'] * events_df['occupancy_rate'] / 100).astype(int)
        events_df['Commission Revenue'] = events_df['Tickets Sold'] * events_df['ticket_price'] * (events_df['commission_rate'] / 100)
        
        st.dataframe(
            events_df[['name', 'ticket_price', 'Tickets Sold', 'commission_rate', 'Commission Revenue', 'start_date']],
            use_container_width=True,
            hide_index=True
        )
        
        # Clear events button
        if st.button("🗑️ Clear All Events"):
            st.session_state.events = []
            st.rerun()
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue Summary", "📈 Projections", "🎯 ROI Analysis", "📊 Comparisons"])
        
        with tab1:
            st.header("Revenue Summary")
            
            # Calculate totals
            total_tickets = sum(event['num_tickets'] for event in st.session_state.events)
            total_tickets_sold = sum(int(event['num_tickets'] * event['occupancy_rate'] / 100) for event in st.session_state.events)
            total_commission_revenue = sum(
                int(event['num_tickets'] * event['occupancy_rate'] / 100) * event['ticket_price'] * (event['commission_rate'] / 100)
                for event in st.session_state.events
            )
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Events", len(st.session_state.events))
            with col2:
                st.metric("Total Tickets", f"{total_tickets:,}")
            with col3:
                st.metric("Expected Sales", f"{total_tickets_sold:,}")
            with col4:
                st.metric("Total Commission Revenue", f"${total_commission_revenue:,.2f}")
            
            # Revenue by event chart
            fig_revenue = px.bar(
                events_df,
                x='name',
                y='Commission Revenue',
                title='Commission Revenue by Event',
                color='commission_rate',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_revenue, use_container_width=True)
        
        with tab2:
            st.header("Revenue Projections")
            
            # Create analyzer instance
            analyzer = MultiEventAnalyzer()
            for event in st.session_state.events:
                analyzer.add_event(
                    event['name'], event['ticket_price'], event['num_tickets'],
                    event['occupancy_rate'], event['commission_rate'],
                    event['start_date'], event['infrastructure_tier']
                )
            
            # Get projections
            projection_months = st.slider("Projection Period (months)", 3, 24, 12)
            projections = analyzer.calculate_monthly_revenue_projection(projection_months)
            
            # Projections chart
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(
                x=projections['Month'],
                y=projections['Revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='green', width=3)
            ))
            fig_proj.add_trace(go.Scatter(
                x=projections['Month'],
                y=projections['Costs'],
                mode='lines+markers',
                name='Costs',
                line=dict(color='red', width=3)
            ))
            fig_proj.add_trace(go.Scatter(
                x=projections['Month'],
                y=projections['Net Profit'],
                mode='lines+markers',
                name='Net Profit',
                line=dict(color='blue', width=3)
            ))
            
            fig_proj.update_layout(
                title='Monthly Revenue Projections',
                xaxis_title='Month',
                yaxis_title='Amount (USD)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_proj, use_container_width=True)
            
            # Projections table
            st.subheader("Monthly Breakdown")
            st.dataframe(projections.round(2), use_container_width=True, hide_index=True)
        
        with tab3:
            st.header("ROI Analysis")
            
            # Calculate ROI for all events
            roi_df = analyzer.calculate_event_roi()
            
            if not roi_df.empty:
                # ROI chart
                fig_roi = px.bar(
                    roi_df,
                    x='Event',
                    y='ROI (%)',
                    title='Return on Investment by Event',
                    color='ROI (%)',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_roi, use_container_width=True)
                
                # ROI table
                st.subheader("Detailed ROI Analysis")
                st.dataframe(roi_df.round(2), use_container_width=True, hide_index=True)
                
                # ROI insights
                avg_roi = roi_df['ROI (%)'].mean()
                best_event = roi_df.loc[roi_df['ROI (%)'].idxmax(), 'Event']
                worst_event = roi_df.loc[roi_df['ROI (%)'].idxmin(), 'Event']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average ROI", f"{avg_roi:.1f}%")
                with col2:
                    st.metric("Best Performing Event", best_event)
                with col3:
                    st.metric("Needs Improvement", worst_event)
        
        with tab4:
            st.header("Event Comparisons")
            
            # Commission rate vs ROI scatter plot
            if len(st.session_state.events) > 1:
                roi_df = analyzer.calculate_event_roi()
                
                fig_scatter = px.scatter(
                    roi_df,
                    x='Commission Rate (%)',
                    y='ROI (%)',
                    size='Commission Revenue',
                    hover_name='Event',
                    title='Commission Rate vs ROI',
                    color='Net Profit',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Event size comparison
                fig_size = px.scatter(
                    events_df,
                    x='num_tickets',
                    y='Commission Revenue',
                    size='occupancy_rate',
                    hover_name='name',
                    title='Event Size vs Revenue',
                    color='commission_rate',
                    color_continuous_scale='viridis'
                )
                fig_size.update_layout(
                    xaxis_title='Number of Tickets',
                    yaxis_title='Commission Revenue (USD)'
                )
                st.plotly_chart(fig_size, use_container_width=True)
            else:
                st.info("Add more events to see comparisons!")

if __name__ == "__main__":
    show_multi_event_analysis() 