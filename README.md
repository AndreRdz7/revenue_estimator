# 🎫 E-Ticketing Platform Revenue Estimator

A comprehensive revenue estimation tool for e-ticketing platforms using MATIC/Polygon blockchain, seats.io for seating management, and multiple payment processors (Stripe, PayPal, MercadoPago).

## 🚀 Features

### Revenue Analysis
- **Commission Rate Optimization**: Analyze revenue across commission rates from 7-15% (customizable)
- **Break-Even Analysis**: Find the minimum commission rate needed to cover all costs
- **Real-time Calculations**: Instant updates as you adjust parameters
- **Interactive Visualizations**: Charts and graphs to understand revenue patterns

### Cost Management
- **Payment Processor Fees**: Accurate calculations for Stripe (2.9% + $0.30), PayPal (3.49% + $0.49), and MercadoPago (3.99% + $0.25)
- **Blockchain Costs**: MATIC minting and transfer fees (~$0.002 per mint + $0.001 per transfer)
- **Seats.io Integration**: Base monthly cost ($50) + per-booking fees ($0.05)
- **AWS Infrastructure**: Four tiers from Básico ($964/month) to Máximo ($6,166/month)

### Event Planning
- **Flexible Event Parameters**: Ticket prices, quantities, occupancy rates
- **Sales Duration**: Factor in time-based costs over sales periods
- **Infrastructure Scaling**: Match your infrastructure tier to expected event load

## 📊 What This Tool Calculates

### Revenue Components
1. **Commission Revenue**: Your platform's commission on each ticket sale
2. **Gross Revenue**: Total revenue including customer payments
3. **Net Revenue**: Profit after all costs and fees

### Cost Components
1. **Payment Processing Fees**: Varies by processor and includes fixed + percentage fees
2. **Blockchain Costs**: MATIC minting and transfer costs per ticket
3. **SaaS Costs**: Seats.io booking fees and base monthly subscription
4. **Infrastructure**: AWS services scaled to your event volume

### Key Metrics
- Break-even commission rate
- Profit margins at different commission levels
- Cost breakdown analysis
- Monthly infrastructure projections

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone or download the project
cd E_Ticketing_Revenue_Estimator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Dependencies
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `plotly`: Interactive charts and visualizations
- `python-dateutil`: Date/time utilities

## 💡 How to Use

### 1. Set Event Parameters
- **Average Ticket Price**: Base price before commission (e.g., $25)
- **Number of Tickets**: Total available tickets (e.g., 500)
- **Occupancy Rate**: Expected percentage of tickets sold (e.g., 80%)
- **Sales Duration**: How long tickets are on sale (e.g., 2 months)

### 2. Configure Infrastructure
- **Infrastructure Tier**: Choose based on expected load:
  - **Básico**: $964/month - Small events, basic load
  - **Moderado**: $1,445/month - Medium events, moderate load
  - **Intenso**: $2,535/month - Large events, high load
  - **Máximo**: $6,166/month - Very large events, maximum load

### 3. Set Payment Distribution
- **Stripe**: Percentage of payments through Stripe
- **PayPal**: Percentage of payments through PayPal
- **MercadoPago**: Automatically calculated remainder

### 4. Analyze Results
- **Revenue Analysis**: See how different commission rates affect your revenue
- **Break-Even Analysis**: Find the minimum viable commission rate
- **Commission Comparison**: Compare specific rates side-by-side
- **Cost Breakdown**: Understand where your money goes

## 📈 Example Scenario

**Event Setup:**
- 500 tickets at $25 each
- 80% occupancy (400 tickets sold)
- 2-month sales duration
- Moderado infrastructure tier
- 10% commission rate

**Results:**
- Final ticket price: $27.50
- Gross revenue: $11,000
- Commission revenue: $1,000
- Total costs: ~$450 (payment fees + blockchain + infrastructure)
- **Net revenue: ~$550**

## 🔧 Service Cost Assumptions

### Payment Processors
- **Stripe**: 2.9% + $0.30 per transaction
- **PayPal**: 3.49% + $0.49 per transaction
- **MercadoPago**: 3.99% + $0.25 per transaction

### Blockchain (MATIC/Polygon)
- **NFT Minting**: ~$0.002 per ticket
- **Transfer Costs**: ~$0.001 per ticket

### Seats.io
- **Base Monthly**: $50
- **Per Booking**: $0.05 per ticket

### AWS Infrastructure
Based on provided cost structure:
- **Básico**: $964/month
- **Moderado**: $1,445/month
- **Intenso**: $2,535/month
- **Máximo**: $6,166/month

## 🎯 Key Insights

### Commission Rate Strategy
- **7-9%**: Competitive but may not cover all costs for smaller events
- **10-12%**: Balanced approach, typically profitable for most event sizes
- **13-15%**: Higher margins but may affect ticket sales competitiveness

### Break-Even Factors
1. **Event Size**: Larger events have better economies of scale
2. **Infrastructure Tier**: Higher tiers increase break-even commission needed
3. **Sales Duration**: Longer sales periods increase infrastructure costs
4. **Payment Mix**: Processor choice significantly affects fees

### Optimization Tips
- Use lower infrastructure tiers for smaller events
- Optimize payment processor mix based on your audience
- Consider volume discounts for multiple events per month
- Factor in marketing and customer acquisition costs

## 🔮 Future Enhancements

- **Multi-Event Planning**: Plan revenue across multiple events
- **Seasonal Analysis**: Factor in seasonal demand patterns
- **Customer Acquisition Costs**: Include marketing and acquisition expenses
- **Dynamic Pricing**: Integrate dynamic pricing strategies
- **Risk Analysis**: Monte Carlo simulations for demand uncertainty

## 📝 Notes

This tool provides estimates based on current market rates and typical service costs. Actual costs may vary based on:
- Volume discounts with service providers
- Contract negotiations
- Geographic location
- Specific service configurations
- Market rate fluctuations

Always verify current pricing with service providers for production planning.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool!

## 📄 License

This project is open source and available under the MIT License. 