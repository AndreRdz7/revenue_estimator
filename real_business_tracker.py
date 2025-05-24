import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import json
import os
from typing import Dict, List, Tuple, Any
# ML imports (optional)
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    
import warnings
warnings.filterwarnings('ignore')

# Constants for business operations
class BusinessConstants:
    # Mexico Tax Rates (approximate for 2024)
    HONORARIOS_ISR_RATE = 0.10  # 10% ISR for professional fees
    IVA_RATE = 0.16  # 16% IVA
    
    # Business savings targets
    SAFETY_FUND_MONTHS = 6  # 6 months of 'Intenso' infrastructure as safety
    INTENSO_MONTHLY_COST = 2535.00  # From infrastructure tiers
    SAFETY_FUND_TARGET = SAFETY_FUND_MONTHS * INTENSO_MONTHLY_COST  # $15,210
    
    # Salary management
    PARTNERS = 2  # Two partners (50/50 split)
    MIN_MONTHLY_SALARY = 1000.00  # Minimum viable salary per person
    
    # Data persistence
    DATA_FILE = "business_data.json"

class RealBusinessTracker:
    def __init__(self):
        self.data = self.load_data()
    
    def load_data(self) -> Dict:
        """Load persistent business data"""
        if os.path.exists(BusinessConstants.DATA_FILE):
            try:
                with open(BusinessConstants.DATA_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'income_records': [],
            'expense_records': [],
            'salary_payments': [],
            'dividend_payments': [],
            'cash_balance': 0.0,
            'last_updated': str(datetime.now())
        }
    
    def save_data(self):
        """Save business data to file"""
        self.data['last_updated'] = str(datetime.now())
        with open(BusinessConstants.DATA_FILE, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def add_income_record(self, amount: float, description: str, date: datetime, 
                         client_name: str = "", invoice_number: str = "", 
                         commission_rate: float = 0.0):
        """Add income record from client payments"""
        record = {
            'id': len(self.data['income_records']) + 1,
            'date': str(date),
            'amount': amount,
            'description': description,
            'client_name': client_name,
            'invoice_number': invoice_number,
            'commission_rate': commission_rate,
            'taxes': {
                'iva_amount': amount * BusinessConstants.IVA_RATE,
                'net_amount': amount / (1 + BusinessConstants.IVA_RATE)
            }
        }
        self.data['income_records'].append(record)
        self.data['cash_balance'] += amount
        self.save_data()
        return record
    
    def add_expense_record(self, amount: float, description: str, date: datetime,
                          category: str = "General", infrastructure_tier: str = None):
        """Add expense record (infrastructure, salaries, etc.)"""
        record = {
            'id': len(self.data['expense_records']) + 1,
            'date': str(date),
            'amount': amount,
            'description': description,
            'category': category,
            'infrastructure_tier': infrastructure_tier
        }
        self.data['expense_records'].append(record)
        self.data['cash_balance'] -= amount
        self.save_data()
        return record
    
    def calculate_available_salary(self) -> Dict:
        """Calculate if we can pay ourselves and how much"""
        current_balance = self.data['cash_balance']
        safety_fund_needed = BusinessConstants.SAFETY_FUND_TARGET
        
        # Calculate monthly expenses (average of last 3 months)
        expense_df = self.get_expenses_dataframe()
        if not expense_df.empty:
            expense_df['date'] = pd.to_datetime(expense_df['date'])
            recent_expenses = expense_df[expense_df['date'] >= (datetime.now() - timedelta(days=90))]
            monthly_expenses = recent_expenses['amount'].sum() / 3 if len(recent_expenses) > 0 else 0
        else:
            monthly_expenses = BusinessConstants.INTENSO_MONTHLY_COST  # Default estimate
        
        # Available for salaries = Current Balance - Safety Fund - Next Month Expenses
        available_for_salaries = current_balance - safety_fund_needed - monthly_expenses
        
        # Per partner (50/50 split)
        per_partner_available = available_for_salaries / BusinessConstants.PARTNERS
        
        can_pay_minimum = per_partner_available >= BusinessConstants.MIN_MONTHLY_SALARY
        
        return {
            'current_balance': current_balance,
            'safety_fund_target': safety_fund_needed,
            'safety_fund_status': current_balance >= safety_fund_needed,
            'monthly_expenses': monthly_expenses,
            'available_for_salaries': available_for_salaries,
            'per_partner_available': per_partner_available,
            'can_pay_minimum': can_pay_minimum,
            'recommended_salary': max(0, per_partner_available) if can_pay_minimum else 0
        }
    
    def calculate_quarterly_dividends(self) -> Dict:
        """Calculate potential quarterly dividend payments"""
        salary_info = self.calculate_available_salary()
        
        # Only consider dividends if safety fund is met and minimum salaries paid
        if not salary_info['safety_fund_status']:
            return {'can_pay_dividends': False, 'reason': 'Safety fund not met'}
        
        if not salary_info['can_pay_minimum']:
            return {'can_pay_dividends': False, 'reason': 'Cannot pay minimum salaries'}
        
        # Calculate quarterly profit (income - expenses for last 3 months)
        income_df = self.get_income_dataframe()
        expense_df = self.get_expenses_dataframe()
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months
        
        quarterly_income = 0
        quarterly_expenses = 0
        
        if not income_df.empty:
            income_df['date'] = pd.to_datetime(income_df['date'])
            recent_income = income_df[income_df['date'] >= start_date]['amount'].sum()
            quarterly_income = recent_income
        
        if not expense_df.empty:
            expense_df['date'] = pd.to_datetime(expense_df['date'])
            recent_expenses = expense_df[expense_df['date'] >= start_date]['amount'].sum()
            quarterly_expenses = recent_expenses
        
        quarterly_profit = quarterly_income - quarterly_expenses
        
        # Conservative approach: only distribute 50% of quarterly profit
        dividend_pool = quarterly_profit * 0.5
        per_partner_dividend = dividend_pool / BusinessConstants.PARTNERS
        
        return {
            'can_pay_dividends': dividend_pool > 0,
            'quarterly_income': quarterly_income,
            'quarterly_expenses': quarterly_expenses,
            'quarterly_profit': quarterly_profit,
            'dividend_pool': dividend_pool,
            'per_partner_dividend': per_partner_dividend
        }
    
    def calculate_mexico_taxes(self, income_amount: float, is_salary: bool = True) -> Dict:
        """Calculate approximate Mexico taxes"""
        if is_salary:  # Honorarios (professional fees)
            # Simplified calculation for honorarios
            net_income = income_amount / (1 + BusinessConstants.IVA_RATE)  # Remove IVA
            isr_tax = net_income * BusinessConstants.HONORARIOS_ISR_RATE
            iva_tax = income_amount - net_income
            
            return {
                'gross_income': income_amount,
                'net_income': net_income,
                'isr_tax': isr_tax,
                'iva_tax': iva_tax,
                'total_taxes': isr_tax + iva_tax,
                'take_home': income_amount - isr_tax - iva_tax
            }
        else:
            # Dividend taxes (different rates apply)
            dividend_tax_rate = 0.10  # Simplified dividend tax rate
            dividend_tax = income_amount * dividend_tax_rate
            
            return {
                'gross_dividend': income_amount,
                'dividend_tax': dividend_tax,
                'net_dividend': income_amount - dividend_tax
            }
    
    def get_income_dataframe(self) -> pd.DataFrame:
        """Get income records as DataFrame"""
        if not self.data['income_records']:
            return pd.DataFrame()
        return pd.DataFrame(self.data['income_records'])
    
    def get_expenses_dataframe(self) -> pd.DataFrame:
        """Get expense records as DataFrame"""
        if not self.data['expense_records']:
            return pd.DataFrame()
        return pd.DataFrame(self.data['expense_records'])
    
    def predict_income_ml(self, months_ahead: int = 6) -> Dict:
        """Use ML to predict future income and salary potential"""
        if not ML_AVAILABLE:
            return {
                'prediction_available': False,
                'reason': 'ML dependencies not installed (pip install scikit-learn)'
            }
        
        income_df = self.get_income_dataframe()
        
        if len(income_df) < 3:  # Need minimum data for ML
            return {
                'prediction_available': False,
                'reason': 'Insufficient data (need at least 3 income records)'
            }
        
        # Prepare data for ML
        income_df['date'] = pd.to_datetime(income_df['date'])
        income_df = income_df.sort_values('date')
        
        # Create features: days since first record, month, quarter
        base_date = income_df['date'].min()
        income_df['days_since_start'] = (income_df['date'] - base_date).dt.days
        income_df['month'] = income_df['date'].dt.month
        income_df['quarter'] = income_df['date'].dt.quarter
        
        # Group by month for monthly predictions
        monthly_data = income_df.groupby(income_df['date'].dt.to_period('M')).agg({
            'amount': 'sum',
            'days_since_start': 'first'
        }).reset_index()
        
        if len(monthly_data) < 2:
            return {
                'prediction_available': False,
                'reason': 'Need at least 2 months of data'
            }
        
        # Prepare ML features
        X = monthly_data[['days_since_start']].values
        y = monthly_data['amount'].values
        
        # Try polynomial features for better fit
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Predict future months
        last_date = income_df['date'].max()
        predictions = []
        
        for i in range(1, months_ahead + 1):
            future_date = last_date + relativedelta(months=i)
            days_since_start = (future_date - base_date).days
            X_future = poly_features.transform([[days_since_start]])
            predicted_amount = model.predict(X_future)[0]
            
            predictions.append({
                'month': future_date.strftime('%Y-%m'),
                'predicted_income': max(0, predicted_amount),  # No negative predictions
                'date': future_date
            })
        
        # Calculate model accuracy
        y_pred = model.predict(X_poly)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Predict salary raise potential
        avg_predicted_monthly = np.mean([p['predicted_income'] for p in predictions])
        current_avg_monthly = income_df.groupby(income_df['date'].dt.to_period('M'))['amount'].sum().mean()
        
        growth_rate = (avg_predicted_monthly - current_avg_monthly) / current_avg_monthly if current_avg_monthly > 0 else 0
        
        return {
            'prediction_available': True,
            'predictions': predictions,
            'model_accuracy': {
                'mae': mae,
                'r2_score': r2,
                'data_points': len(monthly_data)
            },
            'growth_analysis': {
                'current_avg_monthly': current_avg_monthly,
                'predicted_avg_monthly': avg_predicted_monthly,
                'growth_rate': growth_rate,
                'salary_raise_potential': growth_rate > 0.1  # 10% growth suggests raise potential
            }
        }

def show_real_business_tracker():
    """Main UI for the Real Business Tracker"""
    st.title("💼 Real Business Income & Outcome Tracker")
    st.markdown("**Track actual business finances, calculate salaries, dividends, and predict growth**")
    st.markdown("---")
    
    # Initialize tracker
    if 'business_tracker' not in st.session_state:
        st.session_state.business_tracker = RealBusinessTracker()
    
    tracker = st.session_state.business_tracker
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💰 Dashboard", "📥 Income Tracking", "📤 Expense Tracking", 
        "👨‍💼 Salary Management", "📈 ML Predictions", "💾 Data Export"
    ])
    
    with tab1:
        st.header("💰 Business Financial Dashboard")
        
        # Current financial status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_balance = tracker.data['cash_balance']
            st.metric("💵 Current Cash Balance", f"${current_balance:,.2f}")
            
            safety_status = tracker.calculate_available_salary()
            safety_color = "🟢" if safety_status['safety_fund_status'] else "🔴"
            st.metric(f"{safety_color} Safety Fund Status", 
                     f"${BusinessConstants.SAFETY_FUND_TARGET:,.2f}")
        
        with col2:
            total_income = sum(record['amount'] for record in tracker.data['income_records'])
            total_expenses = sum(record['amount'] for record in tracker.data['expense_records'])
            st.metric("📈 Total Income", f"${total_income:,.2f}")
            st.metric("📉 Total Expenses", f"${total_expenses:,.2f}")
        
        with col3:
            net_profit = total_income - total_expenses
            profit_color = "🟢" if net_profit > 0 else "🔴"
            st.metric(f"{profit_color} Net Profit", f"${net_profit:,.2f}")
            
            # Current month income
            current_month_income = 0
            for record in tracker.data['income_records']:
                record_date = datetime.strptime(record['date'], '%Y-%m-%d %H:%M:%S')
                if record_date.month == datetime.now().month and record_date.year == datetime.now().year:
                    current_month_income += record['amount']
            st.metric("📅 This Month Income", f"${current_month_income:,.2f}")
        
        # Salary analysis
        st.markdown("---")
        st.subheader("👨‍💼 Salary Analysis")
        
        salary_info = tracker.calculate_available_salary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if salary_info['can_pay_minimum']:
                st.success(f"✅ **Can pay salaries!**")
                st.success(f"💰 **Recommended per partner: ${salary_info['recommended_salary']:,.2f}**")
            else:
                st.error("❌ **Cannot pay minimum salaries yet**")
                st.error(f"💰 **Need ${BusinessConstants.MIN_MONTHLY_SALARY * BusinessConstants.PARTNERS:,.2f} minimum**")
            
            st.write(f"**Available for salaries:** ${salary_info['available_for_salaries']:,.2f}")
            st.write(f"**Per partner (50/50):** ${salary_info['per_partner_available']:,.2f}")
        
        with col2:
            # Safety fund progress
            safety_progress = min(1.0, current_balance / BusinessConstants.SAFETY_FUND_TARGET)
            st.progress(safety_progress)
            st.write(f"**Safety Fund Progress:** {safety_progress:.1%}")
            st.write(f"**Target:** ${BusinessConstants.SAFETY_FUND_TARGET:,.2f} (6 months Intenso tier)")
            st.write(f"**Current:** ${current_balance:,.2f}")
        
        # Quarterly dividend analysis
        st.markdown("---")
        st.subheader("💎 Quarterly Dividend Analysis")
        
        dividend_info = tracker.calculate_quarterly_dividends()
        
        if dividend_info['can_pay_dividends']:
            st.success(f"✅ **Dividend payment possible!**")
            st.success(f"💰 **Per partner dividend: ${dividend_info['per_partner_dividend']:,.2f}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Quarterly Income:** ${dividend_info['quarterly_income']:,.2f}")
                st.write(f"**Quarterly Expenses:** ${dividend_info['quarterly_expenses']:,.2f}")
            with col2:
                st.write(f"**Quarterly Profit:** ${dividend_info['quarterly_profit']:,.2f}")
                st.write(f"**Dividend Pool (50%):** ${dividend_info['dividend_pool']:,.2f}")
        else:
            st.warning(f"⚠️ **No dividends yet:** {dividend_info['reason']}")
    
    with tab2:
        st.header("📥 Income Tracking")
        st.markdown("**Register payments from clients for events**")
        
        # Add new income
        with st.expander("➕ Add New Income Record", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                income_amount = st.number_input("Income Amount (USD)", min_value=0.0, value=0.0, step=100.0)
                client_name = st.text_input("Client Name")
                invoice_number = st.text_input("Invoice Number")
            
            with col2:
                description = st.text_area("Description", value="Event payment")
                commission_rate = st.slider("Commission Rate (%)", 0.0, 30.0, 10.0)
                income_date = st.date_input("Income Date", value=datetime.now())
            
            if st.button("💰 Add Income Record") and income_amount > 0:
                record = tracker.add_income_record(
                    amount=income_amount,
                    description=description,
                    date=datetime.combine(income_date, datetime.min.time()),
                    client_name=client_name,
                    invoice_number=invoice_number,
                    commission_rate=commission_rate
                )
                st.success(f"✅ Added income record: ${income_amount:,.2f}")
                st.rerun()
        
        # Show income records
        income_df = tracker.get_income_dataframe()
        if not income_df.empty:
            st.subheader("📊 Income Records")
            
            # Format for display
            display_df = income_df.copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Income chart
            income_df['date'] = pd.to_datetime(income_df['date'])
            monthly_income = income_df.groupby(income_df['date'].dt.to_period('M'))['amount'].sum().reset_index()
            monthly_income['date'] = monthly_income['date'].astype(str)
            
            fig = px.bar(monthly_income, x='date', y='amount', 
                        title='Monthly Income Trend',
                        labels={'amount': 'Income (USD)', 'date': 'Month'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📤 Expense Tracking")
        st.markdown("**Track infrastructure costs, salaries, and other business expenses**")
        
        # Add new expense
        with st.expander("➕ Add New Expense Record", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                expense_amount = st.number_input("Expense Amount (USD)", min_value=0.0, value=0.0, step=50.0)
                expense_category = st.selectbox("Category", [
                    "AWS Infrastructure", "Salaries", "Software Licenses", 
                    "Office Expenses", "Marketing", "Professional Services", "Other"
                ])
                infrastructure_tier = st.selectbox("Infrastructure Tier (if applicable)", 
                                                 ["None", "Básico", "Moderado", "Intenso", "Máximo"])
            
            with col2:
                expense_description = st.text_area("Description")
                expense_date = st.date_input("Expense Date", value=datetime.now())
            
            if st.button("💸 Add Expense Record") and expense_amount > 0:
                tier = infrastructure_tier if infrastructure_tier != "None" else None
                record = tracker.add_expense_record(
                    amount=expense_amount,
                    description=expense_description,
                    date=datetime.combine(expense_date, datetime.min.time()),
                    category=expense_category,
                    infrastructure_tier=tier
                )
                st.success(f"✅ Added expense record: ${expense_amount:,.2f}")
                st.rerun()
        
        # Show expense records
        expense_df = tracker.get_expenses_dataframe()
        if not expense_df.empty:
            st.subheader("📊 Expense Records")
            
            # Format for display
            display_df = expense_df.copy()
            display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Expenses by category
            category_expenses = expense_df.groupby('category')['amount'].sum().reset_index()
            fig = px.pie(category_expenses, values='amount', names='category',
                        title='Expenses by Category')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("👨‍💼 Salary Management")
        st.markdown("**Calculate and track partner salaries with Mexico tax considerations**")
        
        salary_info = tracker.calculate_available_salary()
        
        # Current salary status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Current Salary Analysis")
            
            if salary_info['can_pay_minimum']:
                st.success("✅ **Salary payment feasible**")
                recommended_salary = salary_info['recommended_salary']
                
                # Calculate taxes for recommended salary
                tax_info = tracker.calculate_mexico_taxes(recommended_salary, is_salary=True)
                
                st.write(f"**Recommended per partner:** ${recommended_salary:,.2f}")
                st.write(f"**ISR Tax (10%):** ${tax_info['isr_tax']:,.2f}")
                st.write(f"**IVA (16%):** ${tax_info['iva_tax']:,.2f}")
                st.write(f"**Take-home pay:** ${tax_info['take_home']:,.2f}")
                
                # Salary payment simulator
                st.markdown("---")
                st.subheader("💸 Process Salary Payment")
                
                custom_salary = st.number_input(
                    "Salary per partner", 
                    min_value=0.0, 
                    value=recommended_salary, 
                    step=100.0
                )
                
                if st.button("💰 Process Salary Payments") and custom_salary > 0:
                    # Add expense records for both partners
                    today = datetime.now()
                    
                    for partner_num in range(1, BusinessConstants.PARTNERS + 1):
                        record = tracker.add_expense_record(
                            amount=custom_salary,
                            description=f"Salary payment - Partner {partner_num}",
                            date=today,
                            category="Salaries"
                        )
                    
                    st.success(f"✅ Processed salary payments: ${custom_salary * BusinessConstants.PARTNERS:,.2f} total")
                    st.rerun()
            
            else:
                st.error("❌ **Cannot pay minimum salaries**")
                st.write(f"**Current balance:** ${salary_info['current_balance']:,.2f}")
                st.write(f"**Safety fund needed:** ${salary_info['safety_fund_target']:,.2f}")
                st.write(f"**Monthly expenses:** ${salary_info['monthly_expenses']:,.2f}")
                st.write(f"**Shortfall:** ${-salary_info['available_for_salaries']:,.2f}")
        
        with col2:
            st.subheader("🏦 Mexico Tax Calculator")
            
            test_amount = st.number_input("Test Salary Amount", min_value=0.0, value=5000.0, step=100.0)
            
            if test_amount > 0:
                tax_calc = tracker.calculate_mexico_taxes(test_amount, is_salary=True)
                
                st.write("**Tax Breakdown (Honorarios):**")
                st.write(f"• **Gross Amount:** ${tax_calc['gross_income']:,.2f}")
                st.write(f"• **Net (before taxes):** ${tax_calc['net_income']:,.2f}")
                st.write(f"• **ISR (10%):** ${tax_calc['isr_tax']:,.2f}")
                st.write(f"• **IVA (16%):** ${tax_calc['iva_tax']:,.2f}")
                st.write(f"• **Total Taxes:** ${tax_calc['total_taxes']:,.2f}")
                st.write(f"• **Take-home:** ${tax_calc['take_home']:,.2f}")
                
                # Effective tax rate
                effective_rate = (tax_calc['total_taxes'] / test_amount) * 100
                st.write(f"• **Effective Tax Rate:** {effective_rate:.1f}%")
        
        # Salary history
        st.markdown("---")
        st.subheader("📊 Salary Payment History")
        
        salary_records = [record for record in tracker.data['expense_records'] 
                         if record['category'] == 'Salaries']
        
        if salary_records:
            salary_df = pd.DataFrame(salary_records)
            salary_df['date'] = pd.to_datetime(salary_df['date']).dt.strftime('%Y-%m-%d')
            salary_df['amount'] = salary_df['amount'].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(salary_df[['date', 'amount', 'description']], 
                        use_container_width=True, hide_index=True)
        else:
            st.info("💡 No salary payments recorded yet")
    
    with tab5:
        st.header("📈 Machine Learning Predictions")
        st.markdown("**AI-powered income predictions and salary optimization**")
        
        # Get ML predictions
        ml_results = tracker.predict_income_ml(months_ahead=6)
        
        if ml_results['prediction_available']:
            st.success("✅ **ML Model trained successfully!**")
            
            # Model accuracy
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Accuracy")
                accuracy = ml_results['model_accuracy']
                st.write(f"**R² Score:** {accuracy['r2_score']:.3f}")
                st.write(f"**Mean Absolute Error:** ${accuracy['mae']:,.2f}")
                st.write(f"**Data Points:** {accuracy['data_points']}")
                
                # Accuracy interpretation
                if accuracy['r2_score'] > 0.7:
                    st.success("🎯 **Excellent prediction accuracy**")
                elif accuracy['r2_score'] > 0.5:
                    st.warning("⚠️ **Moderate prediction accuracy**")
                else:
                    st.error("❌ **Low prediction accuracy - need more data**")
            
            with col2:
                st.subheader("📈 Growth Analysis")
                growth = ml_results['growth_analysis']
                
                st.write(f"**Current Avg Monthly:** ${growth['current_avg_monthly']:,.2f}")
                st.write(f"**Predicted Avg Monthly:** ${growth['predicted_avg_monthly']:,.2f}")
                
                growth_rate = growth['growth_rate'] * 100
                if growth_rate > 0:
                    st.success(f"📈 **Growth Rate:** +{growth_rate:.1f}%")
                    if growth['salary_raise_potential']:
                        st.success("💰 **Salary raise potential detected!**")
                else:
                    st.warning(f"📉 **Growth Rate:** {growth_rate:.1f}%")
            
            # Predictions chart
            st.subheader("🔮 Income Predictions")
            
            predictions = ml_results['predictions']
            pred_df = pd.DataFrame(predictions)
            
            fig = px.line(pred_df, x='month', y='predicted_income',
                         title='Predicted Monthly Income (Next 6 Months)',
                         labels={'predicted_income': 'Predicted Income (USD)', 'month': 'Month'})
            
            # Add current data for comparison
            income_df = tracker.get_income_dataframe()
            if not income_df.empty:
                income_df['date'] = pd.to_datetime(income_df['date'])
                monthly_actual = income_df.groupby(income_df['date'].dt.to_period('M'))['amount'].sum().reset_index()
                monthly_actual['month'] = monthly_actual['date'].astype(str)
                
                fig.add_scatter(x=monthly_actual['month'], y=monthly_actual['amount'],
                              mode='markers+lines', name='Actual Income',
                              line=dict(color='green'))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Salary recommendations based on predictions
            st.subheader("💰 AI Salary Recommendations")
            
            avg_predicted = np.mean([p['predicted_income'] for p in predictions])
            
            # Conservative salary calculation (30% of predicted income per partner)
            conservative_salary = (avg_predicted * 0.3) / BusinessConstants.PARTNERS
            
            st.write(f"**Based on predicted income of ${avg_predicted:,.2f}/month:**")
            st.write(f"**Conservative salary per partner:** ${conservative_salary:,.2f}")
            
            if conservative_salary > BusinessConstants.MIN_MONTHLY_SALARY:
                st.success("✅ **Predictions support salary increases!**")
            else:
                st.warning("⚠️ **Conservative approach recommends waiting**")
        
        else:
            st.warning(f"⚠️ **ML Predictions not available:** {ml_results['reason']}")
            st.info("💡 Add more income records to enable ML predictions")
    
    with tab6:
        st.header("💾 Data Export & Persistence")
        st.markdown("**Export analysis data and manage persistence**")
        
        # Current data status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Data")
            st.write(f"**Income Records:** {len(tracker.data['income_records'])}")
            st.write(f"**Expense Records:** {len(tracker.data['expense_records'])}")
            st.write(f"**Last Updated:** {tracker.data['last_updated']}")
            st.write(f"**Data File:** {BusinessConstants.DATA_FILE}")
        
        with col2:
            st.subheader("💰 Financial Summary")
            st.write(f"**Cash Balance:** ${tracker.data['cash_balance']:,.2f}")
            
            total_income = sum(r['amount'] for r in tracker.data['income_records'])
            total_expenses = sum(r['amount'] for r in tracker.data['expense_records'])
            
            st.write(f"**Total Income:** ${total_income:,.2f}")
            st.write(f"**Total Expenses:** ${total_expenses:,.2f}")
            st.write(f"**Net Profit:** ${total_income - total_expenses:,.2f}")
        
        # Export options
        st.markdown("---")
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Income CSV"):
                income_df = tracker.get_income_dataframe()
                if not income_df.empty:
                    csv = income_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Income CSV",
                        data=csv,
                        file_name=f"income_records_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📊 Export Expenses CSV"):
                expense_df = tracker.get_expenses_dataframe()
                if not expense_df.empty:
                    csv = expense_df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download Expenses CSV",
                        data=csv,
                        file_name=f"expense_records_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📋 Export Full Report"):
                # Create comprehensive report
                report_data = {
                    'summary': {
                        'total_income': sum(r['amount'] for r in tracker.data['income_records']),
                        'total_expenses': sum(r['amount'] for r in tracker.data['expense_records']),
                        'cash_balance': tracker.data['cash_balance'],
                        'export_date': str(datetime.now())
                    },
                    'salary_analysis': tracker.calculate_available_salary(),
                    'dividend_analysis': tracker.calculate_quarterly_dividends(),
                    'income_records': tracker.data['income_records'],
                    'expense_records': tracker.data['expense_records']
                }
                
                json_str = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="💾 Download Full Report JSON",
                    data=json_str,
                    file_name=f"business_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        # Data management
        st.markdown("---")
        st.subheader("🔧 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data"):
                st.session_state.business_tracker = RealBusinessTracker()
                st.success("✅ Data refreshed from file")
                st.rerun()
        
        with col2:
            if st.button("⚠️ Clear All Data", type="secondary"):
                if st.confirm("Are you sure you want to delete all business data?"):
                    if os.path.exists(BusinessConstants.DATA_FILE):
                        os.remove(BusinessConstants.DATA_FILE)
                    st.session_state.business_tracker = RealBusinessTracker()
                    st.success("✅ All data cleared")
                    st.rerun()
        
        # File persistence info
        st.markdown("---")
        st.subheader("💡 About Data Persistence")
        st.info("""
        **Data Storage:** All income/expense data is automatically saved to `business_data.json`
        
        **Automatic Backup:** Data is saved after every transaction
        
        **Export Options:** Use CSV exports for spreadsheet analysis or JSON for complete backups
        
        **Single/Multi Event Analysis:** These are temporary and don't need persistence - export to PDF when ready
        """)

if __name__ == "__main__":
    show_real_business_tracker() 