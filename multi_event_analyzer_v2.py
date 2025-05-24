import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from aws_infrastructure_advisor import AWSInfrastructureAdvisor
from enhanced_infrastructure_advisor import EnhancedAWSInfrastructureAdvisor

def format_currency(amount):
    """Format currency with consistent $1,000.00 pattern"""
    return f"${amount:,.2f}"

def format_dataframe_currencies(df, currency_columns):
    """Format specified columns in a dataframe as currency"""
    df_formatted = df.copy()
    for col in currency_columns:
        if col in df_formatted.columns:
            df_formatted[col] = df_formatted[col].apply(lambda x: format_currency(x) if isinstance(x, (int, float)) else x)
    return df_formatted

class SharedInfraMultiEventAnalyzer:
    def __init__(self):
        self.events = []
        self.shared_settings = {
            'infrastructure_tier': 'Moderado',
            'campaign_duration_months': 10,
            'stripe_pct': 40,
            'paypal_pct': 35,
            'mercadopago_pct': 25,
            'exclude_salary': False,
            'monthly_platform_fee': 0
        }
    
    def get_service_costs(self):
        """Get service costs from main app"""
        return {
            'STRIPE_RATE': 0.029,
            'STRIPE_FIXED': 0.30,
            'PAYPAL_RATE': 0.0349,
            'PAYPAL_FIXED': 0.49,
            'MERCADOPAGO_RATE': 0.0399,
            'MERCADOPAGO_FIXED': 0.25,
            'MATIC_MINT_COST': 0.002,
            'MATIC_TRANSFER_COST': 0.001,
            'SEATSIO_PER_BOOKING': 0.17,
            'AWS_COSTS': {
                'Básico': 64.00,
                'Moderado': 245.00,
                'Intenso': 735.00,
                'Máximo': 3166.00
            },
            'SALARY_COSTS': {
                'Básico': 900.00,
                'Moderado': 1200.00,
                'Intenso': 1800.00,
                'Máximo': 3000.00
            }
        }
    
    @property
    def infrastructure_costs(self):
        """Get total infrastructure costs per tier (AWS + Salary)"""
        costs = self.get_service_costs()
        return {
            'Básico': costs['AWS_COSTS']['Básico'] + costs['SALARY_COSTS']['Básico'],
            'Moderado': costs['AWS_COSTS']['Moderado'] + costs['SALARY_COSTS']['Moderado'],
            'Intenso': costs['AWS_COSTS']['Intenso'] + costs['SALARY_COSTS']['Intenso'],
            'Máximo': costs['AWS_COSTS']['Máximo'] + costs['SALARY_COSTS']['Máximo']
        }
    
    def calculate_shared_infrastructure_cost(self):
        """Calculate total infrastructure cost for the entire campaign"""
        costs = self.get_service_costs()
        tier = self.shared_settings['infrastructure_tier']
        duration = self.shared_settings['campaign_duration_months']
        
        aws_cost = costs['AWS_COSTS'][tier] * duration
        salary_cost = costs['SALARY_COSTS'][tier] * duration if not self.shared_settings['exclude_salary'] else 0
        platform_fee_revenue = self.shared_settings['monthly_platform_fee'] * duration
        
        return {
            'aws_total': aws_cost,
            'salary_total': salary_cost,
            'infrastructure_total': aws_cost + salary_cost,
            'platform_fee_total': platform_fee_revenue
        }
    
    def calculate_event_revenue_and_costs(self, event, commission_rate=None):
        """Calculate revenue and variable costs for a single event"""
        costs = self.get_service_costs()
        
        # Use provided commission rate or event's rate
        rate = commission_rate if commission_rate is not None else event['commission_rate']
        
        tickets_sold = int(event['num_tickets'] * event['occupancy_rate'] / 100)
        final_ticket_price = event['ticket_price'] * (1 + rate / 100)
        gross_revenue = tickets_sold * final_ticket_price
        commission_revenue = tickets_sold * event['ticket_price'] * (rate / 100)
        
        # Calculate payment processor fees
        stripe_fees = (gross_revenue * self.shared_settings['stripe_pct'] / 100) * costs['STRIPE_RATE'] + costs['STRIPE_FIXED'] if self.shared_settings['stripe_pct'] > 0 else 0
        paypal_fees = (gross_revenue * self.shared_settings['paypal_pct'] / 100) * costs['PAYPAL_RATE'] + costs['PAYPAL_FIXED'] if self.shared_settings['paypal_pct'] > 0 else 0
        mercadopago_fees = (gross_revenue * self.shared_settings['mercadopago_pct'] / 100) * costs['MERCADOPAGO_RATE'] + costs['MERCADOPAGO_FIXED'] if self.shared_settings['mercadopago_pct'] > 0 else 0
        
        payment_fees = stripe_fees + paypal_fees + mercadopago_fees
        
        # Variable costs per event
        matic_costs = tickets_sold * (costs['MATIC_MINT_COST'] + costs['MATIC_TRANSFER_COST'])
        seatsio_costs = tickets_sold * costs['SEATSIO_PER_BOOKING']
        variable_costs = payment_fees + matic_costs + seatsio_costs
        
        return {
            'tickets_sold': tickets_sold,
            'final_ticket_price': final_ticket_price,
            'gross_revenue': gross_revenue,
            'commission_revenue': commission_revenue,
            'payment_fees': payment_fees,
            'matic_costs': matic_costs,
            'seatsio_costs': seatsio_costs,
            'variable_costs': variable_costs,
            'commission_rate': rate
        }
    
    def calculate_cumulative_analysis(self, commission_rate):
        """Calculate cumulative profitability analysis"""
        if not self.events:
            return pd.DataFrame()
        
        infra_costs = self.calculate_shared_infrastructure_cost()
        
        cumulative_data = []
        cumulative_commission = 0
        cumulative_variable_costs = 0
        cumulative_platform_fee = infra_costs['platform_fee_total']
        
        for i, event in enumerate(self.events):
            event_data = self.calculate_event_revenue_and_costs(event, commission_rate)
            
            cumulative_commission += event_data['commission_revenue']
            cumulative_variable_costs += event_data['variable_costs']
            
            total_revenue = cumulative_commission + cumulative_platform_fee
            total_costs = cumulative_variable_costs + infra_costs['infrastructure_total']
            net_profit = total_revenue - total_costs
            
            cumulative_data.append({
                'Event Number': i + 1,
                'Event Name': event['name'],
                'Event Commission': event_data['commission_revenue'],
                'Cumulative Commission': cumulative_commission,
                'Cumulative Platform Fee': cumulative_platform_fee,
                'Cumulative Total Revenue': total_revenue,
                'Cumulative Variable Costs': cumulative_variable_costs,
                'Infrastructure Costs': infra_costs['infrastructure_total'],
                'Cumulative Total Costs': total_costs,
                'Cumulative Net Profit': net_profit,
                'Break Even': 'Yes' if net_profit >= 0 else 'No'
            })
        
        return pd.DataFrame(cumulative_data)
    
    def find_optimal_commission_rate(self):
        """Find the minimum commission rate that makes all events profitable"""
        if not self.events:
            return None
        
        # Test commission rates from 1% to 30%
        for rate in np.arange(1, 31, 0.1):
            cumulative_df = self.calculate_cumulative_analysis(rate)
            if not cumulative_df.empty and cumulative_df.iloc[-1]['Cumulative Net Profit'] >= 0:
                return rate
        
        return None
    
    def calculate_commission_comparison(self, rates_to_compare):
        """Compare different commission rates"""
        comparison_data = []
        
        for rate in rates_to_compare:
            cumulative_df = self.calculate_cumulative_analysis(rate)
            if not cumulative_df.empty:
                final_row = cumulative_df.iloc[-1]
                
                # Find break-even event number
                breakeven_event = None
                for _, row in cumulative_df.iterrows():
                    if row['Cumulative Net Profit'] >= 0:
                        breakeven_event = row['Event Number']
                        break
                
                comparison_data.append({
                    'Commission Rate (%)': rate,
                    'Total Commission Revenue': format_currency(final_row['Cumulative Commission']),
                    'Total Platform Fee': format_currency(final_row['Cumulative Platform Fee']),
                    'Total Revenue': format_currency(final_row['Cumulative Total Revenue']),
                    'Total Costs': format_currency(final_row['Cumulative Total Costs']),
                    'Net Profit': format_currency(final_row['Cumulative Net Profit']),
                    'Break-Even Event #': breakeven_event if breakeven_event else 'Never',
                    'Profitable': 'Yes' if final_row['Cumulative Net Profit'] >= 0 else 'No'
                })
        
        return pd.DataFrame(comparison_data)

def show_shared_multi_event_analysis():
    """Display the new shared infrastructure multi-event analysis"""
    st.title("🏢 Multi-Event Campaign Analysis")
    st.markdown("**Perfect for clients hosting multiple events with shared infrastructure**")
    st.markdown("---")
    
    # Initialize session state
    if 'shared_events' not in st.session_state:
        st.session_state.shared_events = []
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SharedInfraMultiEventAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for shared settings
    st.sidebar.header("🏗️ Shared Infrastructure Settings")
    
    analyzer.shared_settings['infrastructure_tier'] = st.sidebar.selectbox(
        "Infrastructure Tier (Shared)",
        options=['Básico', 'Moderado', 'Intenso', 'Máximo'],
        index=1,
        help="One infrastructure tier for all events"
    )
    
    # Calculate default campaign duration based on events
    default_duration = 10  # fallback
    if st.session_state.shared_events:
        # Get the latest event date
        latest_event_date = max(event['event_date'] for event in st.session_state.shared_events)
        today = datetime.now().date()
        
        # Calculate months difference, rounded up
        months_diff = (latest_event_date.year - today.year) * 12 + (latest_event_date.month - today.month)
        if latest_event_date.day > today.day:
            months_diff += 1
        
        default_duration = max(1, months_diff)  # At least 1 month
    
    analyzer.shared_settings['campaign_duration_months'] = st.sidebar.number_input(
        "Campaign Duration (months)",
        min_value=1,
        max_value=24,
        value=default_duration,
        help="Auto-calculated from latest event date, or set manually"
    )
    
    analyzer.shared_settings['monthly_platform_fee'] = st.sidebar.number_input(
        "Monthly Platform Fee (USD)",
        min_value=0.0,
        value=0.0,
        step=50.0,
        help="Fixed monthly fee charged to client"
    )
    
    analyzer.shared_settings['exclude_salary'] = st.sidebar.checkbox(
        "Exclude Salary Costs",
        value=False
    )
    
    st.sidebar.subheader("Payment Distribution (Shared)")
    analyzer.shared_settings['stripe_pct'] = st.sidebar.slider("Stripe (%)", 0, 100, 40)
    analyzer.shared_settings['paypal_pct'] = st.sidebar.slider("PayPal (%)", 0, 100, 35)
    analyzer.shared_settings['mercadopago_pct'] = 100 - analyzer.shared_settings['stripe_pct'] - analyzer.shared_settings['paypal_pct']
    st.sidebar.write(f"MercadoPago: {analyzer.shared_settings['mercadopago_pct']}%")
    
    # Event input
    with st.expander("➕ Add Event to Campaign", expanded=len(st.session_state.shared_events) == 0):
        col1, col2 = st.columns(2)
        
        with col1:
            # Don't reset event name - keep it persistent
            if 'current_event_name' not in st.session_state:
                st.session_state.current_event_name = f"Event {len(st.session_state.shared_events) + 1}"
            
            event_name = st.text_input("Event Name", value=st.session_state.current_event_name)
            ticket_price = st.number_input("Base Ticket Price ($)", min_value=1.0, value=25.0, step=1.0)
            num_tickets = st.number_input("Number of Tickets", min_value=1, value=500, step=10)
        
        with col2:
            occupancy_rate = st.slider("Expected Occupancy (%)", 10, 100, 80)
            commission_rate = st.slider("Commission Rate (%)", 5, 25, 10)
            event_date = st.date_input("Event Date", value=datetime.now())
        
        if st.button("Add Event to Campaign"):
            new_event = {
                'name': event_name,
                'ticket_price': ticket_price,
                'num_tickets': num_tickets,
                'occupancy_rate': occupancy_rate,
                'commission_rate': commission_rate,
                'event_date': event_date
            }
            st.session_state.shared_events.append(new_event)
            analyzer.events = st.session_state.shared_events
            
            # Update event name for next event
            st.session_state.current_event_name = f"Event {len(st.session_state.shared_events) + 1}"
            
            st.success(f"✅ Added {event_name} to campaign!")
            st.rerun()
    
    # Display events
    if st.session_state.shared_events:
        analyzer.events = st.session_state.shared_events
        
        # Show events summary
        total_tickets = sum(event['num_tickets'] for event in st.session_state.shared_events)
        st.subheader(f"📋 Campaign Overview: {len(st.session_state.shared_events)} Events • {total_tickets:,} Total Tickets")
        
        events_df = pd.DataFrame(st.session_state.shared_events)
        events_df['Tickets Sold'] = (events_df['num_tickets'] * events_df['occupancy_rate'] / 100).astype(int)
        
        # Create display dataframe with proper headers and formatting
        display_df = events_df[['name', 'ticket_price', 'num_tickets', 'occupancy_rate', 'commission_rate', 'event_date']].copy()
        display_df.columns = ['Event Name', 'Ticket Price ($)', 'Total Tickets', 'Occupancy (%)', 'Commission (%)', 'Event Date']
        
        # Format money columns
        display_df['Ticket Price ($)'] = display_df['Ticket Price ($)'].apply(lambda x: f"${x:,.2f}")
        
        # Make table editable
        st.write("**💡 Tip:** Click on any cell below to edit event details")
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",  # Allow adding/deleting rows
            column_config={
                "Event Name": st.column_config.TextColumn("Event Name", max_chars=50),
                "Ticket Price ($)": st.column_config.TextColumn("Ticket Price ($)"),
                "Total Tickets": st.column_config.NumberColumn("Total Tickets", min_value=1, step=1),
                "Occupancy (%)": st.column_config.NumberColumn("Occupancy (%)", min_value=1, max_value=100, step=1),
                "Commission (%)": st.column_config.NumberColumn("Commission (%)", min_value=1, max_value=50, step=1),
                "Event Date": st.column_config.DateColumn("Event Date")
            }
        )
        
        # Update session state if changes were made
        if not edited_df.equals(display_df):
            try:
                # Convert edited dataframe back to original format
                updated_events = []
                for _, row in edited_df.iterrows():
                    # Parse price back to float (remove $ and commas)
                    price_str = str(row['Ticket Price ($)']).replace('$', '').replace(',', '')
                    try:
                        ticket_price = float(price_str)
                    except:
                        ticket_price = 25.0  # Default fallback
                    
                    updated_event = {
                        'name': row['Event Name'],
                        'ticket_price': ticket_price,
                        'num_tickets': int(row['Total Tickets']),
                        'occupancy_rate': int(row['Occupancy (%)']),
                        'commission_rate': int(row['Commission (%)']),
                        'event_date': row['Event Date']
                    }
                    updated_events.append(updated_event)
                
                st.session_state.shared_events = updated_events
                analyzer.events = updated_events
                st.success("✅ Events updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating events: {str(e)}")
        
        # Show calculated metrics
        events_summary_df = pd.DataFrame(st.session_state.shared_events)
        events_summary_df['Tickets Sold'] = (events_summary_df['num_tickets'] * events_summary_df['occupancy_rate'] / 100).astype(int)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_tickets_sold = events_summary_df['Tickets Sold'].sum()
            st.metric("Total Tickets Sold", f"{total_tickets_sold:,}")
        with col2:
            avg_ticket_price = events_summary_df['ticket_price'].mean()
            st.metric("Avg Ticket Price", f"${avg_ticket_price:,.2f}")
        with col3:
            avg_occupancy = events_summary_df['occupancy_rate'].mean()
            st.metric("Avg Occupancy", f"{avg_occupancy:.1f}%")
        
        if st.button("🗑️ Clear All Events"):
            st.session_state.shared_events = []
            analyzer.events = []
            # Reset event name counter
            st.session_state.current_event_name = "Event 1"
            st.rerun()
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Break-Even Analysis", "📊 Commission Comparison", "📈 Cumulative Profitability", "💰 Cost Summary"])
        
        with tab1:
            st.header("🏆 Break-Even Analysis")
            
            # Find optimal commission rate
            optimal_rate = analyzer.find_optimal_commission_rate()
            
            if optimal_rate:
                st.success(f"✅ **Minimum profitable commission rate: {optimal_rate:.1f}%**")
                
                # Show break-even analysis at optimal rate
                cumulative_df = analyzer.calculate_cumulative_analysis(optimal_rate)
                
                # Find break-even event
                breakeven_event = None
                for _, row in cumulative_df.iterrows():
                    if row['Cumulative Net Profit'] >= 0:
                        breakeven_event = row['Event Number']
                        break
                
                if breakeven_event:
                    st.info(f"🎯 **Break-even achieved at Event #{breakeven_event}**")
                    
                    col1, col2, col3 = st.columns(3)
                    final_row = cumulative_df.iloc[-1]
                    
                    with col1:
                        st.metric("Final Net Profit", f"${final_row['Cumulative Net Profit']:,.2f}")
                    with col2:
                        st.metric("Total Revenue", f"${final_row['Cumulative Total Revenue']:,.2f}")
                    with col3:
                        st.metric("Break-Even Event", f"#{breakeven_event}/{len(st.session_state.shared_events)}")
                
                # Chart showing cumulative profitability
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_df['Event Number'],
                    y=cumulative_df['Cumulative Net Profit'],
                    mode='lines+markers',
                    name='Cumulative Net Profit',
                    line=dict(color='blue', width=3)
                ))
                
                # Add break-even line
                fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-Even Line")
                
                # Mark break-even point
                if breakeven_event:
                    breakeven_row = cumulative_df[cumulative_df['Event Number'] == breakeven_event].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=[breakeven_event],
                        y=[breakeven_row['Cumulative Net Profit']],
                        mode='markers',
                        name=f'🏆 Break-Even (Event #{breakeven_event})',
                        marker=dict(color='gold', size=15, symbol='star')
                    ))
                
                fig.update_layout(
                    title=f'Cumulative Profitability at {optimal_rate:.1f}% Commission',
                    xaxis_title='Event Number',
                    yaxis_title='Cumulative Net Profit (USD)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("❌ No profitable commission rate found up to 30%. Consider:")
                st.write("- Reducing infrastructure tier")
                st.write("- Adding monthly platform fee")
                st.write("- Increasing ticket prices or occupancy rates")
        
        with tab2:
            st.header("📊 Commission Rate Comparison")
            
            # Commission rate comparison
            st.subheader("Compare Different Commission Rates")
            rates_to_compare = st.multiselect(
                "Select commission rates to compare",
                options=list(range(5, 21)),
                default=[8, 10, 12, 15]
            )
            
            if rates_to_compare:
                comparison_df = analyzer.calculate_commission_comparison(rates_to_compare)
                
                # Comparison chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Total Revenue',
                    x=comparison_df['Commission Rate (%)'],
                    y=comparison_df['Total Revenue']
                ))
                fig.add_trace(go.Bar(
                    name='Total Costs',
                    x=comparison_df['Commission Rate (%)'],
                    y=comparison_df['Total Costs']
                ))
                fig.add_trace(go.Bar(
                    name='Net Profit',
                    x=comparison_df['Commission Rate (%)'],
                    y=comparison_df['Net Profit']
                ))
                
                fig.update_layout(
                    title='Revenue vs Costs by Commission Rate',
                    xaxis_title='Commission Rate (%)',
                    yaxis_title='Amount (USD)',
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison table
                st.dataframe(comparison_df.round(2), use_container_width=True, hide_index=True)
        
        with tab3:
            st.header("📈 Cumulative Profitability")
            
            # Select commission rate for detailed analysis
            analysis_rate = st.slider("Commission Rate for Analysis (%)", 5, 25, 10)
            
            cumulative_df = analyzer.calculate_cumulative_analysis(analysis_rate)
            
            if not cumulative_df.empty:
                # Cumulative chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_df['Event Number'],
                    y=cumulative_df['Cumulative Total Revenue'],
                    mode='lines+markers',
                    name='Cumulative Revenue',
                    line=dict(color='green', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=cumulative_df['Event Number'],
                    y=cumulative_df['Cumulative Total Costs'],
                    mode='lines+markers',
                    name='Cumulative Costs',
                    line=dict(color='red', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=cumulative_df['Event Number'],
                    y=cumulative_df['Cumulative Net Profit'],
                    mode='lines+markers',
                    name='Cumulative Net Profit',
                    line=dict(color='blue', width=3)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title=f'Cumulative Analysis at {analysis_rate}% Commission',
                    xaxis_title='Event Number',
                    yaxis_title='Amount (USD)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Event-by-Event Analysis")
                
                # Format currency columns in the cumulative analysis
                currency_columns = [
                    'Event Commission', 'Cumulative Commission', 'Cumulative Platform Fee',
                    'Cumulative Total Revenue', 'Cumulative Variable Costs', 'Infrastructure Costs',
                    'Cumulative Total Costs', 'Cumulative Net Profit'
                ]
                cumulative_df_formatted = format_dataframe_currencies(cumulative_df, currency_columns)
                st.dataframe(cumulative_df_formatted, use_container_width=True, hide_index=True)
        
        with tab4:
            st.header("💰 Cost Summary")
            
            infra_costs = analyzer.calculate_shared_infrastructure_cost()
            
            st.subheader("Shared Infrastructure Costs")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("AWS Infrastructure", f"${infra_costs['aws_total']:,.2f}")
                st.write(f"({analyzer.shared_settings['campaign_duration_months']} months)")
            
            with col2:
                if not analyzer.shared_settings['exclude_salary']:
                    st.metric("Staff Salary", f"${infra_costs['salary_total']:,.2f}")
                else:
                    st.metric("Staff Salary", "Excluded")
            
            with col3:
                st.metric("Total Infrastructure", f"${infra_costs['infrastructure_total']:,.2f}")
            
            if analyzer.shared_settings['monthly_platform_fee'] > 0:
                st.metric("Platform Fee Revenue", f"${infra_costs['platform_fee_total']:,.2f}")
            
            # 🖥️ SELF-HOSTED REDIS COST ANALYSIS
            st.markdown("---")
            st.subheader("🖥️ Self-Hosted Redis vs AWS ElastiCache Analysis")
            st.markdown("**💡 Considering hardware in Puebla, Mexico for cost savings**")
            
            # Estimate cache requirements based on events
            total_tickets = sum(event['num_tickets'] for event in st.session_state.shared_events)
            estimated_cache_gb = max(1, total_tickets / 10000)  # Rough estimate
            
            # Create enhanced advisor for Redis analysis
            redis_advisor = EnhancedAWSInfrastructureAdvisor()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📊 Estimated Cache Needs: {estimated_cache_gb:.1f} GB**")
                st.markdown(f"**🎫 Based on {total_tickets:,} total tickets across all events**")
                
                # Generate Redis comparison
                redis_comparison = redis_advisor.generate_redis_cost_comparison(estimated_cache_gb)
                
                if not redis_comparison.empty:
                    st.markdown("**💰 Cost Comparison:**")
                    st.dataframe(redis_comparison, use_container_width=True, hide_index=True)
            
            with col2:
                # Detailed savings analysis
                campaign_months = analyzer.shared_settings['campaign_duration_months']
                savings_analysis = redis_advisor.calculate_redis_savings_analysis(estimated_cache_gb, campaign_months)
                
                if savings_analysis['total_savings'] > 0:
                    st.success(f"""
                    🎯 **Potential Savings: ${savings_analysis['total_savings']:,.2f}**  
                    📅 **Over {campaign_months} months**  
                    📈 **{savings_analysis['savings_percentage']:.1f}% cost reduction**  
                    ⚖️ **Break-even: {savings_analysis['break_even_months']:.1f} months**
                    """)
                    
                    st.markdown("**🖥️ Recommended Self-Hosted Option:**")
                    option = savings_analysis['self_hosted_option']
                    st.markdown(f"• **{option['name']}**")
                    st.markdown(f"• **Specs:** {option['specs']}")
                    st.markdown(f"• **Hardware Cost:** ${option['hardware_cost_usd']:,.2f}")
                    st.markdown(f"• **Monthly Electricity:** ${option['monthly_electricity_cost']:,.2f}")
                    st.markdown(f"• **Total Monthly:** ${option['monthly_total_cost']:,.2f}")
                    
                    st.markdown("**☁️ AWS ElastiCache Alternative:**")
                    aws_option = savings_analysis['aws_option']
                    st.markdown(f"• **Memory:** {aws_option['memory_gb']} GB")
                    st.markdown(f"• **Monthly Cost:** ${aws_option['cost_monthly']:,.2f}")
                else:
                    st.info("💡 **AWS ElastiCache** is more cost-effective for your current usage pattern")
                    st.markdown("**☁️ AWS Benefits:**")
                    st.markdown("• Managed service (no maintenance)")
                    st.markdown("• High availability")
                    st.markdown("• Automatic scaling")
                    st.markdown("• Professional support")
            
            # Considerations section
            with st.expander("⚖️ Self-Hosted vs Cloud Considerations"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Self-Hosted Advantages:**")
                    st.markdown("• Lower long-term operational costs")
                    st.markdown("• Complete control over configuration")
                    st.markdown("• One-time hardware investment")
                    st.markdown("• No vendor lock-in")
                    st.markdown("• Cheap electricity in Puebla (~$0.08/kWh)")
                    st.markdown("• Low cooling requirements (cache server)")
                    
                    st.markdown("**⚠️ Self-Hosted Considerations:**")
                    st.markdown("• Initial hardware investment required")
                    st.markdown("• Setup and configuration time (1-2 days)")
                    st.markdown("• Maintenance responsibility")
                    st.markdown("• Single point of failure (cache only)")
                    st.markdown("• Need basic server administration skills")
                
                with col2:
                    st.markdown("**✅ AWS ElastiCache Advantages:**")
                    st.markdown("• Instant deployment (minutes)")
                    st.markdown("• Fully managed service")
                    st.markdown("• High availability with Multi-AZ")
                    st.markdown("• Automatic backups and updates")
                    st.markdown("• Professional AWS support")
                    st.markdown("• Easy scaling up/down")
                    
                    st.markdown("**⚠️ AWS ElastiCache Considerations:**")
                    st.markdown("• Higher ongoing monthly costs")
                    st.markdown("• Vendor lock-in")
                    st.markdown("• Data transfer costs")
                    st.markdown("• Less control over configuration")
                    st.markdown("• Costs scale with usage/time")
                
                st.markdown("---")
                st.markdown("**🎯 Recommendation Summary:**")
                if savings_analysis['total_savings'] > 0:
                    st.markdown(f"• For campaigns **longer than {savings_analysis['break_even_months']:.1f} months**, self-hosted is more cost-effective")
                    st.markdown("• Self-hosted works well since cache failure doesn't break the system (just slower)")
                    st.markdown("• Puebla's cheap electricity and low cooling needs make hardware very cost-effective")
                else:
                    st.markdown("• For short campaigns, AWS ElastiCache provides better value")
                    st.markdown("• Consider self-hosted for long-term or recurring campaigns")
            
            # Per-event variable costs summary
            st.subheader("Variable Costs per Event")
            if analyzer.events:
                sample_event = analyzer.events[0]  # Use first event as example
                event_data = analyzer.calculate_event_revenue_and_costs(sample_event, 10)  # 10% example
                
                st.write(f"**Example (based on {sample_event['name']} at 10% commission):**")
                st.write(f"- Payment processor fees: ~${event_data['payment_fees']:,.2f}")
                st.write(f"- MATIC blockchain costs: ${event_data['matic_costs']:,.2f}")
                st.write(f"- Seats.io costs: ${event_data['seatsio_costs']:,.2f}")
                st.write(f"- **Total variable costs: ${event_data['variable_costs']:,.2f}**")
            
            # 🚀 ENHANCED SMART INFRASTRUCTURE RECOMMENDATION SECTION
            st.markdown("---")
            st.subheader("🏗️ Enhanced Smart Infrastructure Recommendation")
            
            # Infrastructure optimization options
            col1, col2 = st.columns(2)
            with col1:
                use_enhanced_advisor = st.checkbox("🚀 Use Enhanced Analysis", value=True, help="Advanced operational analysis with custom configurations")
            with col2:
                latency_priority = st.checkbox("⚡ Latency Optimization Priority", value=False, help="Prioritize low latency over cost optimization")
            
            # Initialize infrastructure advisor
            if use_enhanced_advisor:
                advisor = EnhancedAWSInfrastructureAdvisor()
                recommendation = advisor.recommend_infrastructure(st.session_state.shared_events, latency_priority=latency_priority)
            else:
                advisor = AWSInfrastructureAdvisor()
                recommendation = advisor.recommend_infrastructure_tier(st.session_state.shared_events)
            
            # Display recommendation summary
            if use_enhanced_advisor and recommendation['type'] == 'custom':
                # Enhanced advisor with custom configuration
                config = recommendation['config']
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    confidence_color = {
                        'High': '🟢',
                        'Medium': '🟡', 
                        'Low': '🔴'
                    }
                    confidence_level = recommendation['confidence'].split(' - ')[0].strip()
                    
                    st.markdown(f"""
                    **🎯 Configuration:** {config['tier_name']} (Optimized)  
                    **📊 Confidence:** {confidence_color.get(confidence_level, '⚪')} {recommendation['confidence']}  
                    **💰 Monthly Cost:** ${config['total_monthly_cost']:,.2f}  
                    **📝 Description:** {config['description']}
                    """)
                
                with col2:
                    aws_savings = config['total_aws_cost']
                    current_tier_cost = analyzer.infrastructure_costs[analyzer.shared_settings['infrastructure_tier']]
                    cost_diff = config['total_monthly_cost'] - current_tier_cost
                    
                    if abs(cost_diff) < 50:
                        st.info("📊 Similar cost to current")
                    elif cost_diff > 0:
                        st.warning(f"⬆️ Custom config\n+${cost_diff:,.2f}/month")
                    else:
                        st.success(f"⬇️ Cost optimized\n${abs(cost_diff):,.2f}/month savings")
                
                with col3:
                    if st.button("📋 View Details"):
                        st.session_state['show_enhanced_details'] = True
                    
                    if st.button("🔄 Use Custom Config"):
                        # For custom configs, we'll approximate to closest predefined tier
                        closest_tier = "Moderado"  # Default fallback
                        analyzer.shared_settings['infrastructure_tier'] = closest_tier
                        st.success(f"Infrastructure approximated to {closest_tier} tier!")
                        st.rerun()
            
            else:
                # Legacy advisor or predefined tier from enhanced advisor
                if use_enhanced_advisor:
                    # Enhanced advisor but using predefined tier
                    tier_name = recommendation['tier_name']
                    config = recommendation['config']
                    monthly_cost = config['total_cost']
                    description = config['specs'].get('description', f"Predefined {config['name']} tier") if 'specs' in config else "Predefined tier"
                else:
                    # Legacy advisor
                    tier_name = recommendation['recommended_tier']
                    monthly_cost = recommendation['tier_info']['total_cost']
                    description = recommendation['tier_info']['description']
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    confidence_color = {
                        'High': '🟢',
                        'Medium': '🟡', 
                        'Low': '🔴'
                    }
                    confidence_level = recommendation['confidence'].split('(')[0].strip() if '(' in recommendation['confidence'] else recommendation['confidence'].split(' - ')[0].strip()
                    
                    st.markdown(f"""
                    **🎯 Recommended Tier:** {tier_name}  
                    **📊 Confidence:** {confidence_color.get(confidence_level, '⚪')} {recommendation['confidence']}  
                    **💰 Monthly Cost:** ${monthly_cost:,.2f}  
                    **📝 Description:** {description}
                    """)
                
                with col2:
                    current_tier = analyzer.shared_settings['infrastructure_tier']
                    if current_tier == tier_name:
                        st.success("✅ Current tier is optimal!")
                    else:
                        cost_diff = monthly_cost - analyzer.infrastructure_costs[current_tier]
                        if cost_diff > 0:
                            st.warning(f"⬆️ Upgrade needed\n+${cost_diff:,.2f}/month")
                        else:
                            st.info(f"⬇️ Can downgrade\n${abs(cost_diff):,.2f}/month savings")
                
                with col3:
                    if st.button("🔄 Apply Recommendation"):
                        analyzer.shared_settings['infrastructure_tier'] = tier_name
                        st.success(f"Infrastructure updated to {tier_name}!")
                        st.rerun()
            
            # Enhanced Details Section (for custom configurations)
            if use_enhanced_advisor and recommendation['type'] == 'custom' and st.session_state.get('show_enhanced_details', False):
                st.markdown("---")
                st.subheader("🔬 Detailed Operational Analysis")
                
                config = recommendation['config']
                analysis = recommendation['analysis']
                operations = analysis['operations']
                requirements = analysis['requirements']
                
                # Operational Phases Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🎫 Ticket Purchase Phase**")
                    purchase_ops = operations['ticket_purchase_phase']
                    st.markdown(f"• Peak Concurrent Buyers: **{purchase_ops['peak_concurrent_buyers']:,}**")
                    st.markdown(f"• DB Writes/Hour: **{purchase_ops['db_writes_per_hour']:,}**")
                    st.markdown(f"• Payment Transactions/Hour: **{purchase_ops['payment_transactions_hour']:,}**")
                    st.markdown(f"• MATIC Operations: **{purchase_ops['matic_minting_operations']:,}**")
                    
                    st.markdown("**📧 QR & Email Phase**")
                    qr_ops = operations['qr_email_phase']
                    st.markdown(f"• QR Codes to Generate: **{qr_ops['qr_generations_needed']:,}**")
                    st.markdown(f"• Emails to Send: **{qr_ops['emails_to_send']:,}**")
                    st.markdown(f"• Batch Processing Hours: **{qr_ops['batch_processing_hours']:.1f}**")
                    st.markdown(f"• Storage for QRs: **{qr_ops['storage_for_qrs_gb']:.2f} GB**")
                
                with col2:
                    st.markdown("**🔍 Event Day Scanning**")
                    scan_ops = operations['event_day_scanning']
                    st.markdown(f"• Peak Scans/Hour: **{scan_ops['peak_scans_per_hour']:,}**")
                    st.markdown(f"• DB Reads/Hour: **{scan_ops['db_reads_per_hour']:,}**")
                    st.markdown(f"• Concurrent Scanners: **{scan_ops['concurrent_scanners']:,}**")
                    st.markdown(f"• Latency Requirement: **{scan_ops['latency_requirement_ms']} ms**")
                    
                    st.markdown("**👥 Browsing Phase**")
                    browse_ops = operations['browsing_phase']
                    st.markdown(f"• Peak Browsers: **{browse_ops['peak_browsers']:,}**")
                    st.markdown(f"• CDN Requests/Hour: **{browse_ops['cdn_requests_hour']:,}**")
                    st.markdown(f"• Cache Usage: **{browse_ops['cache_usage_gb']:.1f} GB**")
                
                # AWS Components Recommendation
                st.markdown("**🏗️ Custom AWS Architecture**")
                components = config['aws_components']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**💾 Database (RDS)**")
                    rds_info = components['rds']
                    st.markdown(f"Instance: **{rds_info['instance']}**")
                    st.markdown(f"Cost: **${rds_info['cost']:,.2f}/month**")
                    
                    # Get RDS specs for details
                    if use_enhanced_advisor:
                        rds_specs = advisor.aws_services['rds_postgres'][rds_info['instance']]
                        st.markdown(f"• vCPU: {rds_specs['vcpu']}")
                        st.markdown(f"• Memory: {rds_specs['memory_gb']} GB")
                        st.markdown(f"• Max Connections: {rds_specs['max_connections']:,}")
                
                with col2:
                    st.markdown("**🖥️ Compute (EC2)**")
                    ec2_info = components['ec2']
                    st.markdown(f"Instance: **{ec2_info['instance']}**")
                    st.markdown(f"Cost: **${ec2_info['cost']:,.2f}/month**")
                    
                    # Get EC2 specs for details
                    if use_enhanced_advisor:
                        ec2_specs = advisor.aws_services['ec2_compute'][ec2_info['instance']]
                        st.markdown(f"• vCPU: {ec2_specs['vcpu']}")
                        st.markdown(f"• Memory: {ec2_specs['memory_gb']} GB")
                        st.markdown(f"• Concurrent Users: {ec2_specs['concurrent_users']:,}")
                
                with col3:
                    st.markdown("**⚡ Cache (Redis)**")
                    redis_info = components['redis']
                    st.markdown(f"Instance: **{redis_info['instance']}**")
                    st.markdown(f"Cost: **${redis_info['cost']:,.2f}/month**")
                    
                    # Get Redis specs for details
                    if use_enhanced_advisor:
                        redis_specs = advisor.aws_services['elasticache_redis'][redis_info['instance']]
                        st.markdown(f"• vCPU: {redis_specs['vcpu']}")
                        st.markdown(f"• Memory: {redis_specs['memory_gb']} GB")
                
                # Additional Services Cost Breakdown
                st.markdown("**💰 Additional Services**")
                additional_services = components['additional_services']
                
                services_data = []
                for service, cost in additional_services.items():
                    if service != 'total' and cost > 0:
                        service_names = {
                            'ses': 'SES Email Service',
                            'waf': 'Web Application Firewall',
                            's3': 'S3 Object Storage',
                            'cloudfront': 'CloudFront CDN',
                            'route53': 'Route53 DNS',
                            'lambda': 'Lambda Functions',
                            'netlify': 'Netlify Frontend Hosting'
                        }
                        services_data.append({
                            'Service': service_names.get(service, service.upper()),
                            'Monthly Cost': f"${cost:,.2f}",
                            'Category': 'External' if service == 'netlify' else 'AWS'
                        })
                
                if services_data:
                    services_df = pd.DataFrame(services_data)
                    st.dataframe(services_df, use_container_width=True, hide_index=True)
                
                # Cost Drivers & Optimization Notes
                if analysis.get('cost_drivers'):
                    st.markdown("**🎯 Cost Drivers Identified**")
                    for driver in analysis['cost_drivers']:
                        st.markdown(f"• {driver}")
                
                if config.get('optimization_notes'):
                    st.markdown("**💡 Optimization Recommendations**")
                    for note in config['optimization_notes']:
                        st.markdown(f"• {note}")
                
                # Latency Optimization
                if recommendation.get('latency_optimization'):
                    st.markdown("**⚡ Latency Optimization Suggestions**")
                    for suggestion in recommendation['latency_optimization']:
                        st.markdown(f"• {suggestion}")
                
                # Cost Comparison with Predefined Tiers
                if recommendation.get('cost_comparison'):
                    st.markdown("**📊 Cost Comparison with Standard Tiers**")
                    comparison = recommendation['cost_comparison']
                    comparison_data = []
                    
                    for tier_name, data in comparison.items():
                        comparison_data.append({
                            'Tier': tier_name,
                            'Monthly Cost': f"${data['monthly_cost']:,.2f}",
                            'Savings vs Custom': f"${data['savings_vs_custom']:,.2f}",
                            'Fits Requirements': '✅' if data['fits_requirements'] else '❌'
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                if st.button("🔽 Hide Enhanced Details"):
                    st.session_state['show_enhanced_details'] = False
                    st.rerun()
            
            # Technical Analysis Details
            with st.expander("🔧 Technical Analysis Details"):
                if use_enhanced_advisor:
                    if recommendation['type'] == 'custom':
                        st.info("💡 **Enhanced analysis available above!** Click 'View Details' for comprehensive operational breakdown.")
                        requirements = recommendation['analysis']['requirements']
                    else:
                        # Enhanced advisor but predefined tier
                        requirements = recommendation['analysis']['requirements'] if 'analysis' in recommendation else {}
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📈 Campaign Requirements:**")
                        st.markdown(f"• **Events:** {requirements.get('total_events', 0)}")
                        st.markdown(f"• **Total Tickets:** {requirements.get('total_tickets', 0):,}")
                        st.markdown(f"• **Expected Sold:** {requirements.get('total_sold_tickets', 0):,}")
                        st.markdown(f"• **Peak Concurrent Users:** {requirements.get('peak_concurrent_users', 0):,}")
                        st.markdown(f"• **Peak DB Connections:** {requirements.get('peak_db_connections', 0):.0f}")
                        st.markdown(f"• **Storage Needs:** {requirements.get('storage_requirements_gb', 0):.1f} GB")
                        st.markdown(f"• **Email Volume:** {requirements.get('email_volume', 0):,}")
                    
                    with col2:
                        if recommendation['type'] == 'custom':
                            config = recommendation['config']
                            components = config['aws_components']
                            st.markdown("**🛠️ Custom AWS Components:**")
                            st.markdown(f"• **RDS:** {components['rds']['instance']} (${components['rds']['cost']:,.2f}/month)")
                            st.markdown(f"• **EC2:** {components['ec2']['instance']} (${components['ec2']['cost']:,.2f}/month)")
                            st.markdown(f"• **Redis:** {components['redis']['instance']} (${components['redis']['cost']:,.2f}/month)")
                            st.markdown(f"• **Additional Services:** ${components['additional_services']['total']:,.2f}/month")
                            st.markdown(f"**Total AWS: ${config['total_aws_cost']:,.2f}/month**")
                        else:
                            # Predefined tier from enhanced advisor
                            st.markdown("**🛠️ Predefined Tier Selected:**")
                            config = recommendation['config']
                            st.markdown(f"• **Tier:** {config['name']}")
                            st.markdown(f"• **AWS Cost:** ${config['aws_cost']:,.2f}/month")
                            st.markdown(f"• **Salary Cost:** ${config['specs'].get('salary_cost', 0):,.2f}/month")
                            st.markdown(f"**Total Cost: ${config['total_cost']:,.2f}/month**")
                
                else:
                    # Legacy advisor
                    req = recommendation['requirements']
                    analysis = recommendation['analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📈 Campaign Requirements:**")
                        st.markdown(f"• **Events:** {req.get('total_events', 0)}")
                        st.markdown(f"• **Total Tickets:** {req.get('total_tickets', 0):,}")
                        st.markdown(f"• **Expected Sold:** {req.get('total_sold_tickets', 0):,}")
                        st.markdown(f"• **Peak Concurrent Users:** {req.get('peak_concurrent_users', 0):,}")
                        st.markdown(f"• **Peak DB Transactions/hr:** {req.get('peak_db_transactions', 0):,}")
                        st.markdown(f"• **Storage Needs:** {req.get('estimated_storage_gb', 0):.1f} GB")
                    
                    with col2:
                        if analysis:
                            st.markdown("**🛠️ Recommended AWS Components:**")
                            st.markdown(f"• **RDS Instance:** {analysis['recommended_rds']}")
                            st.markdown(f"  - vCPU: {analysis['rds_specs']['vcpu']}, Memory: {analysis['rds_specs']['memory_gb']} GB")
                            st.markdown(f"  - Max Connections: {analysis['rds_specs']['max_connections']:,}")
                            st.markdown(f"• **EC2 Instance:** {analysis['recommended_ec2']}")
                            st.markdown(f"  - vCPU: {analysis['ec2_specs']['vcpu']}, Memory: {analysis['ec2_specs']['memory_gb']} GB")
                            st.markdown(f"  - Max Concurrent Users: {analysis['ec2_specs']['concurrent_users']:,}")
                            
                            st.markdown("**💵 AWS Cost Breakdown:**")
                            costs = analysis['estimated_costs']
                            st.markdown(f"• RDS: ${costs['rds_monthly']:,.2f}/month")
                            st.markdown(f"• EC2: ${costs['ec2_monthly']:,.2f}/month")
                            st.markdown(f"• Storage: ${costs['storage_monthly']:,.2f}/month")
                            st.markdown(f"• Backup: ${costs['backup_monthly']:,.2f}/month")
                            st.markdown(f"• Data Transfer: ${costs['data_transfer_monthly']:,.2f}/month")
                            st.markdown(f"**Total AWS: ${costs['total_aws_monthly']:,.2f}/month**")
            
            # Comprehensive Infrastructure Analysis
            if use_enhanced_advisor:
                with st.expander("🔄 Standard vs Enhanced Comparison"):
                    st.markdown("**Compare Enhanced Analysis with Standard Recommendations**")
                    
                    # Get both recommendations for comparison
                    standard_advisor = AWSInfrastructureAdvisor()
                    standard_recommendation = standard_advisor.recommend_infrastructure_tier(st.session_state.shared_events)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔧 Standard Analysis**")
                        st.markdown(f"• **Recommended Tier:** {standard_recommendation['recommended_tier']}")
                        st.markdown(f"• **Monthly Cost:** ${standard_recommendation['tier_info']['total_cost']:,.2f}")
                        st.markdown(f"• **Confidence:** {standard_recommendation['confidence']}")
                        
                        if standard_recommendation.get('analysis'):
                            analysis = standard_recommendation['analysis']['estimated_costs']
                            st.markdown("**Cost Breakdown:**")
                            st.markdown(f"• RDS: ${analysis['rds_monthly']:,.2f}")
                            st.markdown(f"• EC2: ${analysis['ec2_monthly']:,.2f}")
                            st.markdown(f"• Storage: ${analysis['storage_monthly']:,.2f}")
                    
                    with col2:
                        st.markdown("**🚀 Enhanced Analysis**")
                        if recommendation['type'] == 'custom':
                            config = recommendation['config']
                            st.markdown(f"• **Configuration:** {config['tier_name']} (Custom)")
                            st.markdown(f"• **Monthly Cost:** ${config['total_monthly_cost']:,.2f}")
                            st.markdown(f"• **Confidence:** {recommendation['confidence']}")
                            
                            components = config['aws_components']
                            st.markdown("**Cost Breakdown:**")
                            st.markdown(f"• RDS: ${components['rds']['cost']:,.2f}")
                            st.markdown(f"• EC2: ${components['ec2']['cost']:,.2f}")
                            st.markdown(f"• Redis: ${components['redis']['cost']:,.2f}")
                            st.markdown(f"• Additional: ${components['additional_services']['total']:,.2f}")
                        else:
                            config = recommendation['config']
                            st.markdown(f"• **Recommended Tier:** {config['name']}")
                            st.markdown(f"• **Monthly Cost:** ${config['total_cost']:,.2f}")
                            st.markdown(f"• **Confidence:** {recommendation['confidence']}")
                    
                    # Comparison summary
                    if recommendation['type'] == 'custom':
                        cost_diff = recommendation['config']['total_monthly_cost'] - standard_recommendation['tier_info']['total_cost']
                        if cost_diff < -100:
                            st.success(f"🎯 **Enhanced analysis saves ${abs(cost_diff):,.2f}/month** with custom configuration!")
                        elif cost_diff > 100:
                            st.warning(f"⚠️ Enhanced analysis suggests ${cost_diff:,.2f}/month more for better performance/reliability")
                        else:
                            st.info("📊 Both analyses suggest similar cost ranges")
            
            # Infrastructure Tier Comparison
            with st.expander("📊 Infrastructure Tier Comparison"):
                if use_enhanced_advisor and hasattr(advisor, 'predefined_tiers'):
                    # Enhanced advisor - show custom comparison
                    st.markdown("**Predefined Tiers vs Your Requirements**")
                    
                    if recommendation.get('analysis') and recommendation['analysis'].get('requirements'):
                        requirements = recommendation['analysis']['requirements']
                        
                        comparison_data = []
                        aws_cost_mapping = {'Básico': 64, 'Moderado': 245, 'Intenso': 735, 'Máximo': 3166}
                        
                        for tier_name, tier_specs in advisor.predefined_tiers.items():
                            fits_events = requirements['total_events'] <= tier_specs['max_events']
                            fits_users = requirements['peak_concurrent_users'] <= tier_specs['max_concurrent_users']
                            fits_transactions = requirements.get('peak_db_connections', 0) <= tier_specs.get('max_db_transactions_hour', 0) / 100
                            
                            fits_all = fits_events and fits_users and fits_transactions
                            total_cost = aws_cost_mapping[tier_name] + tier_specs['salary_cost']
                            
                            is_recommended = (tier_name == recommendation.get('tier_name')) or \
                                           (recommendation['type'] == 'predefined' and tier_name == recommendation['config']['name'])
                            
                            comparison_data.append({
                                'Tier': tier_name,
                                'Monthly Cost': f"${total_cost:,.2f}",
                                'AWS Cost': f"${aws_cost_mapping[tier_name]:,.2f}",
                                'Max Events': tier_specs['max_events'],
                                'Max Users': f"{tier_specs['max_concurrent_users']:,}",
                                'Target Latency': f"{tier_specs['target_latency_ms']}ms",
                                'Fits Requirements': '✅' if fits_all else '❌',
                                'Recommended': '🌟' if is_recommended else ''
                            })
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Show custom recommendation if available
                        if recommendation['type'] == 'custom':
                            st.markdown("---")
                            st.markdown("**🎯 Custom Configuration (Recommended)**")
                            custom_config = recommendation['config']
                            
                            custom_data = [{
                                'Configuration': 'Custom Optimized',
                                'Monthly Cost': f"${custom_config['total_monthly_cost']:,.2f}",
                                'AWS Cost': f"${custom_config['total_aws_cost']:,.2f}",
                                'Events Supported': custom_config['performance_profile']['max_events'],
                                'Max Users': f"{custom_config['performance_profile']['max_concurrent_users']:,}",
                                'Target Latency': f"{custom_config['performance_profile']['latency_target_ms']}ms",
                                'Status': '🚀 Optimized'
                            }]
                            
                            custom_df = pd.DataFrame(custom_data)
                            st.dataframe(custom_df, use_container_width=True, hide_index=True)
                    
                else:
                    # Legacy advisor comparison
                    comparison = advisor.generate_cost_comparison(st.session_state.shared_events)
                    
                    comparison_data = []
                    for tier_name, tier_data in comparison.items():
                        fits_all = all(tier_data['fits_requirements'].values())
                        
                        comparison_data.append({
                            'Tier': tier_name,
                            'Monthly Cost': f"${tier_data['total_cost_monthly']:,.2f}",
                            'AWS Cost': f"${tier_data['aws_cost_monthly']:,.2f}",
                            'Max Events': tier_data['capacity']['max_events'],
                            'Max Users': f"{tier_data['capacity']['max_concurrent_users']:,}",
                            'Max Transactions': f"{tier_data['capacity']['max_db_transactions']:,}",
                            'Fits Requirements': '✅' if fits_all else '❌',
                            'Recommended': '🌟' if tier_data['is_recommended'] else ''
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Optimization Recommendations
            with st.expander("💡 Optimization Recommendations"):
                if use_enhanced_advisor:
                    if recommendation['type'] == 'custom':
                        config = recommendation['config']
                        
                        # Show optimization notes from custom config
                        if config.get('optimization_notes'):
                            st.markdown("**🎯 Operational Optimizations:**")
                            for i, note in enumerate(config['optimization_notes'], 1):
                                st.markdown(f"{i}. {note}")
                        
                        # Show latency optimizations
                        if recommendation.get('latency_optimization'):
                            st.markdown("**⚡ Latency Optimizations:**")
                            for i, opt in enumerate(recommendation['latency_optimization'], 1):
                                st.markdown(f"{i}. {opt}")
                        
                        # Cost optimization suggestions
                        st.markdown("**💰 Cost Optimization Tips:**")
                        total_aws_cost = config['total_aws_cost']
                        if total_aws_cost < 200:
                            st.markdown("• Consider Aurora Serverless for variable workloads")
                            st.markdown("• Use Spot instances for non-critical processing")
                        elif total_aws_cost > 500:
                            st.markdown("• Consider Reserved Instances for 20-40% savings")
                            st.markdown("• Review auto-scaling policies to optimize usage")
                        
                        st.markdown("• Schedule QR generation during low-traffic hours")
                        st.markdown("• Use SES bulk email sending to reduce costs")
                        st.markdown("• Consider CDN caching for static assets")
                        
                        # Redis Cost Analysis for Custom Configurations
                        if recommendation['type'] == 'custom':
                            st.markdown("---")
                            st.markdown("**💡 Redis Cost Optimization Analysis**")
                            
                            analysis = recommendation['analysis']
                            cache_memory_needed = analysis['requirements']['cache_memory_gb']
                            
                            # Generate Redis comparison
                            redis_comparison = advisor.generate_redis_cost_comparison(cache_memory_needed)
                            
                            if not redis_comparison.empty:
                                st.markdown(f"**🔍 Cache Memory Needed: {cache_memory_needed:.1f} GB**")
                                st.dataframe(redis_comparison, use_container_width=True, hide_index=True)
                                
                                # Detailed savings analysis
                                savings_analysis = advisor.calculate_redis_savings_analysis(
                                    cache_memory_needed, 
                                    analyzer.shared_settings['campaign_duration_months']
                                )
                                
                                if savings_analysis['total_savings'] > 0:
                                    st.success(f"""
                                    🎯 **Self-Hosted Redis could save ${savings_analysis['total_savings']:,.2f}** 
                                    over {savings_analysis['campaign_duration_months']} months 
                                    ({savings_analysis['savings_percentage']:.1f}% savings)
                                    
                                    💰 **Break-even point:** {savings_analysis['break_even_months']:.1f} months
                                    """)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**🖥️ Self-Hosted Option:**")
                                        option = savings_analysis['self_hosted_option']
                                        st.markdown(f"• **Hardware:** {option['specs']}")
                                        st.markdown(f"• **Initial Cost:** ${option['hardware_cost_usd']:,.2f}")
                                        st.markdown(f"• **Monthly Electricity:** ${option['monthly_electricity_cost']:,.2f}")
                                        st.markdown(f"• **Total Monthly:** ${option['monthly_total_cost']:,.2f}")
                                    
                                    with col2:
                                        st.markdown("**☁️ AWS ElastiCache:**")
                                        aws_option = savings_analysis['aws_option']
                                        st.markdown(f"• **Memory:** {aws_option['memory_gb']} GB")
                                        st.markdown(f"• **Monthly Cost:** ${aws_option['cost_monthly']:,.2f}")
                                        st.markdown(f"• **Setup:** Instant")
                                        st.markdown(f"• **Maintenance:** AWS Managed")
                                else:
                                    st.info("AWS ElastiCache is more cost-effective for your usage pattern")
                            
                            st.markdown("**⚖️ Considerations:**")
                            st.markdown("• ✅ **Self-hosted:** Lower long-term costs, full control, one-time hardware investment")
                            st.markdown("• ⚠️ **Self-hosted:** Setup time, maintenance responsibility, single point of failure")
                            st.markdown("• ✅ **AWS:** Managed service, high availability, instant scaling")
                            st.markdown("• ⚠️ **AWS:** Higher ongoing costs, vendor lock-in")
                    
                    else:
                        # Enhanced advisor but predefined tier
                        if recommendation.get('latency_optimization'):
                            st.markdown("**⚡ Latency Optimizations:**")
                            for i, opt in enumerate(recommendation['latency_optimization'], 1):
                                st.markdown(f"{i}. {opt}")
                        
                        # General recommendations
                        st.markdown("**🔧 General Recommendations:**")
                        st.markdown("• Enable Redis caching for ticket validation")
                        st.markdown("• Use CloudFront CDN for static content delivery")
                        st.markdown("• Consider read replicas for database performance")
                        st.markdown("• Implement batch processing for QR generation")
                
                else:
                    # Legacy advisor recommendations
                    if recommendation.get('recommendations'):
                        st.markdown("**🔧 Infrastructure Recommendations:**")
                        for i, rec in enumerate(recommendation['recommendations'], 1):
                            st.markdown(f"{i}. {rec}")
                    
                    # Add general e-ticketing optimizations
                    st.markdown("**🎫 E-Ticketing Specific Optimizations:**")
                    st.markdown("• Pre-generate QR codes during off-peak hours")
                    st.markdown("• Cache frequently accessed event data")
                    st.markdown("• Use email queues for bulk ticket sending")
                    st.markdown("• Implement progressive web app for better mobile performance")
            
            # Performance Requirements vs Tier Capacity Chart
            with st.expander("📈 Performance Requirements Analysis"):
                # Handle different recommendation structures
                if use_enhanced_advisor:
                    if recommendation.get('analysis') and recommendation['analysis'].get('requirements'):
                        req = recommendation['analysis']['requirements']
                        if recommendation['type'] == 'custom':
                            tier_info = recommendation['config']['performance_profile']
                        else:
                            tier_info = recommendation['config']['specs']
                    else:
                        req = None
                        tier_info = None
                else:
                    req = recommendation.get('requirements')
                    tier_info = recommendation.get('tier_info')
                
                if req and tier_info:
                    # Create performance comparison chart
                    metrics = ['Events', 'Concurrent Users', 'DB Transactions']
                    requirements_values = [
                        req.get('total_events', 0),
                        req.get('peak_concurrent_users', 0),
                        req.get('peak_db_connections', 0) * 100  # Convert connections to transactions estimate
                    ]
                    
                    # Handle different tier_info structures
                    if use_enhanced_advisor and recommendation['type'] == 'custom':
                        capacity_values = [
                            tier_info.get('max_events', 0),
                            tier_info.get('max_concurrent_users', 0),
                            req.get('peak_db_connections', 0) * 150  # Show capacity with buffer
                        ]
                        tier_name = "Custom Configuration"
                    else:
                        capacity_values = [
                            tier_info.get('max_events', 0),
                            tier_info.get('max_concurrent_users', 0),
                            tier_info.get('max_db_transactions', tier_info.get('max_db_transactions_hour', 0))
                        ]
                        tier_name = tier_info.get('name', recommendation.get('recommended_tier', 'Selected Tier'))
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Your Requirements',
                        x=metrics,
                        y=requirements_values,
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name=f'{tier_name} Capacity',
                        x=metrics,
                        y=capacity_values,
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title="Requirements vs Recommended Tier Capacity",
                        barmode='group',
                        yaxis_title="Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Add your first event to start the analysis!")

if __name__ == "__main__":
    show_shared_multi_event_analysis() 