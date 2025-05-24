import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Page configuration
st.set_page_config(
    page_title="E-Ticketing Revenue Estimator",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for external service costs
class ServiceCosts:
    # Payment processor fees (percentage + fixed fee)
    STRIPE_RATE = 0.029  # 2.9% + $0.30
    STRIPE_FIXED = 0.30
    
    PAYPAL_RATE = 0.0349  # 3.49% + $0.49
    PAYPAL_FIXED = 0.49
    
    MERCADOPAGO_RATE = 0.0399  # 3.99% + varies by country
    MERCADOPAGO_FIXED = 0.25
    
    # MATIC/Polygon costs (estimated)
    MATIC_MINT_COST = 0.002  # ~$0.002 per NFT mint
    MATIC_TRANSFER_COST = 0.001  # ~$0.001 per transfer
    
    # Seats.io pricing (corrected based on actual pricing)
    # $410 yearly for up to 2,400 seats = $0.17 per ticket (no additional monthly fee)
    SEATSIO_PER_BOOKING = 0.17  # $0.17 per ticket (includes all seats.io costs)
    
    # Salary costs (Nóminas from original table - optional)
    SALARY_COSTS = {
        'Básico': 900.00,
        'Moderado': 1200.00,
        'Intenso': 1800.00,
        'Máximo': 3000.00
    }
    
    # AWS Infrastructure costs (TOTAL minus Nóminas from user's table)
    AWS_COSTS = {
        'Básico': 64.00,      # $964 - $900 = $64
        'Moderado': 245.00,   # $1445 - $1200 = $245
        'Intenso': 735.00,    # $2535 - $1800 = $735
        'Máximo': 3166.00     # $6166 - $3000 = $3166
    }

def calculate_payment_processor_fee(amount, processor):
    """Calculate payment processor fees"""
    if processor == "Stripe":
        return amount * ServiceCosts.STRIPE_RATE + ServiceCosts.STRIPE_FIXED
    elif processor == "PayPal":
        return amount * ServiceCosts.PAYPAL_RATE + ServiceCosts.PAYPAL_FIXED
    elif processor == "MercadoPago":
        return amount * ServiceCosts.MERCADOPAGO_RATE + ServiceCosts.MERCADOPAGO_FIXED
    return 0

def calculate_monthly_infrastructure_cost(tier, events_per_month, include_salary=True):
    """Calculate monthly infrastructure costs"""
    base_aws = ServiceCosts.AWS_COSTS[tier]
    salary_monthly = ServiceCosts.SALARY_COSTS[tier] if include_salary else 0
    # Note: Seats.io costs are per-ticket only ($0.17 per ticket, no monthly fee)
    return base_aws + salary_monthly

def calculate_per_ticket_costs(num_tickets, processor):
    """Calculate per-ticket variable costs"""
    # MATIC costs (mint + potential transfer)
    matic_cost = ServiceCosts.MATIC_MINT_COST + ServiceCosts.MATIC_TRANSFER_COST
    
    # Seats.io per booking cost
    seatsio_cost = ServiceCosts.SEATSIO_PER_BOOKING
    
    return matic_cost + seatsio_cost

def main():
    st.title("🎫 E-Ticketing Platform Revenue Estimator")
    st.markdown("---")
    
    # Sidebar for main parameters
    st.sidebar.header("Event Parameters")
    
    # Event details
    avg_ticket_price = st.sidebar.number_input(
        "Average Ticket Price (USD)", 
        min_value=1.0, 
        value=25.0, 
        step=1.0,
        help="Base price of tickets before commission"
    )
    
    num_tickets = st.sidebar.number_input(
        "Number of Tickets", 
        min_value=1, 
        value=500, 
        step=10,
        help="Total tickets available for the event"
    )
    
    occupancy_rate = st.sidebar.slider(
        "Expected Occupancy Rate (%)", 
        min_value=10, 
        max_value=100, 
        value=80,
        help="Percentage of tickets expected to be sold"
    )
    
    # Duration settings
    st.sidebar.subheader("Sales Duration")
    sales_duration_months = st.sidebar.number_input(
        "Sales Duration (months)", 
        min_value=0.1, 
        value=2.0, 
        step=0.1,
        help="How long tickets will be on sale"
    )
    
    # Infrastructure tier
    infrastructure_tier = st.sidebar.selectbox(
        "Infrastructure Tier",
        options=list(ServiceCosts.AWS_COSTS.keys()),
        index=1,
        help="AWS infrastructure tier based on expected load"
    )
    
    # Additional Revenue Options
    st.sidebar.subheader("Additional Revenue & Costs")
    monthly_platform_fee = st.sidebar.number_input(
        "Monthly Platform Fee (USD)", 
        min_value=0.0, 
        value=0.0, 
        step=10.0,
        help="Optional monthly fee charged to event organizers for using the platform"
    )
    
    exclude_salary = st.sidebar.checkbox(
        "Exclude Salary Costs", 
        value=False,
        help="Exclude staff salary costs (Nóminas) from infrastructure calculations"
    )
    
    # Payment processor distribution
    st.sidebar.subheader("Payment Processor Distribution")
    stripe_pct = st.sidebar.slider("Stripe (%)", 0, 100, 40)
    paypal_pct = st.sidebar.slider("PayPal (%)", 0, 100, 35)
    mercadopago_pct = 100 - stripe_pct - paypal_pct
    st.sidebar.write(f"MercadoPago: {mercadopago_pct}%")
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue Analysis", "📊 Break-Even Analysis", "📈 Commission Comparison", "🔧 Cost Breakdown"])
    
    with tab1:
        st.header("Revenue Analysis")
        
        # Commission rate analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Commission Rate Range")
            min_commission = st.slider("Minimum Commission (%)", 5, 20, 5)
            max_commission = st.slider("Maximum Commission (%)", 5, 25, 25)
        
        with col2:
            st.subheader("Event Metrics")
            tickets_sold = int(num_tickets * occupancy_rate / 100)
            st.metric("Tickets to Sell", tickets_sold)
            st.metric("Gross Revenue (before commission)", f"${tickets_sold * avg_ticket_price:,.2f}")
            if monthly_platform_fee > 0:
                platform_fee_for_duration = monthly_platform_fee * sales_duration_months
                st.metric("Platform Fee Revenue", f"${platform_fee_for_duration:,.2f}")
        
        # Calculate revenue for different commission rates
        commission_rates = np.arange(min_commission, max_commission + 1, 1)
        revenue_data = []
        
        for rate in commission_rates:
            # Calculate final ticket price (base + commission)
            final_ticket_price = avg_ticket_price * (1 + rate / 100)
            
            # Calculate gross revenue
            gross_revenue = tickets_sold * final_ticket_price
            
            # Calculate platform commission revenue
            commission_revenue = tickets_sold * avg_ticket_price * (rate / 100)
            
            # Calculate payment processor fees
            stripe_fees = calculate_payment_processor_fee(gross_revenue * stripe_pct / 100, "Stripe") if stripe_pct > 0 else 0
            paypal_fees = calculate_payment_processor_fee(gross_revenue * paypal_pct / 100, "PayPal") if paypal_pct > 0 else 0
            mercadopago_fees = calculate_payment_processor_fee(gross_revenue * mercadopago_pct / 100, "MercadoPago") if mercadopago_pct > 0 else 0
            
            total_payment_fees = stripe_fees + paypal_fees + mercadopago_fees
            
            # Calculate per-ticket costs
            per_ticket_cost = calculate_per_ticket_costs(tickets_sold, "Mixed")
            total_variable_costs = per_ticket_cost * tickets_sold
            
            # Calculate monthly infrastructure costs
            monthly_infra_cost = calculate_monthly_infrastructure_cost(infrastructure_tier, 1, not exclude_salary)
            total_infra_cost = monthly_infra_cost * sales_duration_months
            
            # Add monthly platform fee revenue
            platform_fee_revenue = monthly_platform_fee * sales_duration_months
            
            # Net revenue calculation (commission + platform fee - all costs)
            net_revenue = commission_revenue + platform_fee_revenue - total_payment_fees - total_variable_costs - total_infra_cost
            
            revenue_data.append({
                'Commission Rate (%)': rate,
                'Final Ticket Price': final_ticket_price,
                'Gross Revenue': gross_revenue,
                'Commission Revenue': commission_revenue,
                'Platform Fee Revenue': platform_fee_revenue,
                'Total Revenue': commission_revenue + platform_fee_revenue,
                'Payment Processor Fees': total_payment_fees,
                'Variable Costs': total_variable_costs,
                'Infrastructure Costs': total_infra_cost,
                'Net Revenue': net_revenue,
                'Profit Margin (%)': (net_revenue / (commission_revenue + platform_fee_revenue) * 100) if (commission_revenue + platform_fee_revenue) > 0 else 0
            })
        
        # Create DataFrame and display results
        df_revenue = pd.DataFrame(revenue_data)
        
        # Revenue chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Scatter(
            x=df_revenue['Commission Rate (%)'],
            y=df_revenue['Commission Revenue'],
            mode='lines+markers',
            name='Commission Revenue',
            line=dict(color='green', width=3)
        ))
        if monthly_platform_fee > 0:
            fig_revenue.add_trace(go.Scatter(
                x=df_revenue['Commission Rate (%)'],
                y=df_revenue['Total Revenue'],
                mode='lines+markers',
                name='Total Revenue (Commission + Platform Fee)',
                line=dict(color='darkgreen', width=3)
            ))
        fig_revenue.add_trace(go.Scatter(
            x=df_revenue['Commission Rate (%)'],
            y=df_revenue['Net Revenue'],
            mode='lines+markers',
            name='Net Revenue',
            line=dict(color='blue', width=3)
        ))
        
        # Add golden break-even marker! 🏆
        breakeven_rates = df_revenue[df_revenue['Net Revenue'] >= 0]['Commission Rate (%)']
        if not breakeven_rates.empty:
            breakeven_rate = breakeven_rates.iloc[0]
            breakeven_row = df_revenue[df_revenue['Commission Rate (%)'] == breakeven_rate].iloc[0]
            
            fig_revenue.add_trace(go.Scatter(
                x=[breakeven_rate],
                y=[breakeven_row['Net Revenue']],
                mode='markers',
                name=f'🏆 Break-Even Point ({breakeven_rate}%)',
                marker=dict(
                    color='gold',
                    size=15,
                    symbol='star',
                    line=dict(color='darkgoldenrod', width=2)
                )
            ))
        fig_revenue.update_layout(
            title='Revenue vs Commission Rate',
            xaxis_title='Commission Rate (%)',
            yaxis_title='Revenue (USD)',
            hovermode='x unified'
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Revenue table
        st.subheader("Detailed Revenue Analysis")
        st.dataframe(
            df_revenue.round(2),
            use_container_width=True,
            hide_index=True
        )
    
    with tab2:
        st.header("Break-Even Analysis")
        
        # Find break-even point
        breakeven_rates = []
        for _, row in df_revenue.iterrows():
            if row['Net Revenue'] >= 0:
                breakeven_rates.append(row['Commission Rate (%)'])
        
        if breakeven_rates:
            min_breakeven = min(breakeven_rates)
            st.success(f"✅ Break-even achieved at {min_breakeven}% commission rate")
            
            # Show break-even metrics
            breakeven_row = df_revenue[df_revenue['Commission Rate (%)'] == min_breakeven].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Break-even Commission", f"{min_breakeven}%")
            with col2:
                st.metric("Final Ticket Price", f"${breakeven_row['Final Ticket Price']:.2f}")
            with col3:
                st.metric("Commission Revenue", f"${breakeven_row['Commission Revenue']:,.2f}")
            with col4:
                st.metric("Net Revenue", f"${breakeven_row['Net Revenue']:,.2f}")
        else:
            st.error("❌ Break-even not achieved within the commission range. Consider:")
            st.write("- Increasing commission rates")
            st.write("- Reducing infrastructure tier")
            st.write("- Increasing ticket sales")
            st.write("- Reducing sales duration")
    
    with tab3:
        st.header("Commission Rate Comparison")
        
        # Interactive commission rate selector
        # Create smart defaults that are actually available in the options (8-15% range)
        available_rates = commission_rates.tolist()
        default_rates = []
        for rate in [8, 10, 12, 15]:
            if rate in available_rates:
                default_rates.append(rate)
        
        # If we don't have enough defaults, add some available rates
        if len(default_rates) < 2 and len(available_rates) >= 2:
            for rate in available_rates:
                if rate not in default_rates:
                    default_rates.append(rate)
                if len(default_rates) >= 4:
                    break
        
        selected_rates = st.multiselect(
            "Select commission rates to compare",
            options=available_rates,
            default=default_rates[:4]  # Limit to 4 defaults max
        )
        
        if selected_rates:
            comparison_df = df_revenue[df_revenue['Commission Rate (%)'].isin(selected_rates)]
            
            # Create comparison chart
            bars_data = [
                go.Bar(name='Commission Revenue', x=comparison_df['Commission Rate (%)'], y=comparison_df['Commission Revenue']),
            ]
            
            if monthly_platform_fee > 0:
                bars_data.append(go.Bar(name='Platform Fee Revenue', x=comparison_df['Commission Rate (%)'], y=comparison_df['Platform Fee Revenue']))
            
            bars_data.extend([
                go.Bar(name='Total Costs', x=comparison_df['Commission Rate (%)'], 
                      y=comparison_df['Payment Processor Fees'] + comparison_df['Variable Costs'] + comparison_df['Infrastructure Costs']),
                go.Bar(name='Net Revenue', x=comparison_df['Commission Rate (%)'], y=comparison_df['Net Revenue'])
            ])
            
            fig_comparison = go.Figure(data=bars_data)
            
            fig_comparison.update_layout(
                title='Revenue and Cost Comparison by Commission Rate',
                xaxis_title='Commission Rate (%)',
                yaxis_title='Amount (USD)',
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Profit margin chart
            fig_margin = px.line(
                comparison_df, 
                x='Commission Rate (%)', 
                y='Profit Margin (%)',
                title='Profit Margin by Commission Rate',
                markers=True
            )
            st.plotly_chart(fig_margin, use_container_width=True)
    
    with tab4:
        st.header("Cost Breakdown Analysis")
        
        # Select a commission rate for detailed analysis
        available_rates = commission_rates.tolist()
        default_index = min(3, len(available_rates) - 1) if len(available_rates) > 0 else 0
        
        analysis_rate = st.selectbox(
            "Select commission rate for detailed cost analysis",
            options=available_rates,
            index=default_index  # Safe default index
        )
        
        analysis_row = df_revenue[df_revenue['Commission Rate (%)'] == analysis_rate].iloc[0]
        
        # Cost breakdown pie chart
        costs = {
            'Payment Processor Fees': analysis_row['Payment Processor Fees'],
            'MATIC & Blockchain Costs': tickets_sold * (ServiceCosts.MATIC_MINT_COST + ServiceCosts.MATIC_TRANSFER_COST),
            'Seats.io Costs': tickets_sold * ServiceCosts.SEATSIO_PER_BOOKING,
            'AWS Infrastructure': ServiceCosts.AWS_COSTS[infrastructure_tier] * sales_duration_months,
        }
        
        if not exclude_salary:
            costs['Staff Salary (Nóminas)'] = ServiceCosts.SALARY_COSTS[infrastructure_tier] * sales_duration_months
        
        fig_pie = px.pie(
            values=list(costs.values()),
            names=list(costs.keys()),
            title=f'Cost Breakdown at {analysis_rate}% Commission Rate'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Detailed cost table
        st.subheader("Detailed Cost Analysis")
        
        # Calculate individual cost components
        aws_cost = ServiceCosts.AWS_COSTS[infrastructure_tier] * sales_duration_months
        salary_cost = ServiceCosts.SALARY_COSTS[infrastructure_tier] * sales_duration_months if not exclude_salary else 0
        
        cost_data = [
            {'Cost Category': 'Payment Processor Fees', 'Amount': f"${analysis_row['Payment Processor Fees']:,.2f}"},
            {'Cost Category': 'MATIC Minting/Transfer', 'Amount': f"${tickets_sold * (ServiceCosts.MATIC_MINT_COST + ServiceCosts.MATIC_TRANSFER_COST):,.2f}"},
            {'Cost Category': 'Seats.io Costs', 'Amount': f"${tickets_sold * ServiceCosts.SEATSIO_PER_BOOKING:,.2f}"},
            {'Cost Category': 'AWS Infrastructure', 'Amount': f"${aws_cost:,.2f}"},
        ]
        
        if not exclude_salary:
            cost_data.append({'Cost Category': 'Staff Salary (Nóminas)', 'Amount': f"${salary_cost:,.2f}"})
        
        # Add revenue if platform fee exists
        if monthly_platform_fee > 0:
            cost_data.insert(0, {'Cost Category': 'Platform Fee Revenue', 'Amount': f"+${analysis_row['Platform Fee Revenue']:,.2f}"})
            cost_data.insert(0, {'Cost Category': 'Commission Revenue', 'Amount': f"+${analysis_row['Commission Revenue']:,.2f}"})
        
        cost_data.append({'Cost Category': 'Total Costs', 'Amount': f"${analysis_row['Payment Processor Fees'] + analysis_row['Variable Costs'] + analysis_row['Infrastructure Costs']:,.2f}"})
        
        cost_df = pd.DataFrame(cost_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
        
        # Monthly cost projection
        st.subheader("Monthly Cost Projection")
        monthly_costs = calculate_monthly_infrastructure_cost(infrastructure_tier, 1, not exclude_salary)
        st.metric("Monthly Infrastructure Cost", f"${monthly_costs:,.2f}")
        st.write(f"- AWS Services: ${ServiceCosts.AWS_COSTS[infrastructure_tier]:,.2f}")
        if not exclude_salary:
            st.write(f"- Staff Salary (Nóminas): ${ServiceCosts.SALARY_COSTS[infrastructure_tier]:,.2f}")
        else:
            st.write("- Staff Salary (Nóminas): **Excluded**")
        
        st.write("**Note:** Seats.io costs are $0.17 per ticket (no monthly base fee)")
        
        if monthly_platform_fee > 0:
            st.subheader("Additional Revenue")
            st.metric("Monthly Platform Fee", f"${monthly_platform_fee:,.2f}")
            st.write("This fee is charged to event organizers in addition to commission")

# Import multi-event analyzers
try:
    from multi_event_analyzer import show_multi_event_analysis
    MULTI_EVENT_AVAILABLE = True
except ImportError:
    MULTI_EVENT_AVAILABLE = False

try:
    from multi_event_analyzer_v2 import show_shared_multi_event_analysis
    SHARED_MULTI_EVENT_AVAILABLE = True
except ImportError:
    SHARED_MULTI_EVENT_AVAILABLE = False

try:
    from real_business_tracker import show_real_business_tracker
    REAL_BUSINESS_AVAILABLE = True
except ImportError:
    REAL_BUSINESS_AVAILABLE = False

if __name__ == "__main__":
    # Add navigation sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    
    analysis_options = ["Single Event Analysis"]
    if SHARED_MULTI_EVENT_AVAILABLE:
        analysis_options.append("Multi-Event Campaign")
    if REAL_BUSINESS_AVAILABLE:
        analysis_options.append("Real Income/Outcome")
    
    page = st.sidebar.radio("Select Analysis Type", analysis_options)
    
    if page == "Single Event Analysis":
        main()
    elif page == "Multi-Event Campaign" and SHARED_MULTI_EVENT_AVAILABLE:
        show_shared_multi_event_analysis()
    elif page == "Real Income/Outcome" and REAL_BUSINESS_AVAILABLE:
        show_real_business_tracker()
    else:
        main() 