import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Tuple
import math

class TemporalDemandModeler:
    """Models how ticket demand changes over time from release to event day"""
    
    def __init__(self):
        self.demand_phases = {
            'launch_peak': {'duration_weeks': 2, 'intensity': 1.0, 'description': 'Initial ticket release rush'},
            'steady_decline': {'duration_weeks': 8, 'intensity': 0.3, 'description': 'Gradual decline in demand'},
            'pre_event_surge': {'duration_weeks': 3, 'intensity': 0.7, 'description': 'Last-minute surge before event'},
            'event_day_peak': {'duration_days': 1, 'intensity': 2.0, 'description': 'Maximum QR scanning load'},
        }
    
    def calculate_demand_curve(self, start_date: date, event_date: date, tickets_sold: int) -> pd.DataFrame:
        """Generate demand curve from start date to event date"""
        timeline = []
        current_date = start_date
        
        # Calculate total campaign duration
        total_days = (event_date - start_date).days
        
        # Adjust phase durations based on campaign length
        if total_days < 30:  # Short campaign
            phases = {
                'launch_peak': {'duration_weeks': 1, 'intensity': 1.2},
                'steady_decline': {'duration_weeks': max(1, total_days // 7 - 2), 'intensity': 0.4},
                'pre_event_surge': {'duration_weeks': 1, 'intensity': 0.8},
                'event_day_peak': {'duration_days': 1, 'intensity': 2.0},
            }
        else:  # Normal/long campaign
            phases = self.demand_phases.copy()
        
        # Generate daily demand
        while current_date <= event_date:
            days_to_event = (event_date - current_date).days
            phase_info = self._determine_phase(current_date, start_date, event_date, phases)
            
            # Calculate base demand for this day
            base_demand = self._calculate_daily_demand(
                current_date, start_date, event_date, tickets_sold, phase_info
            )
            
            # Add day-of-week modulation (weekends are 20% higher)
            weekday_multiplier = 1.2 if current_date.weekday() >= 5 else 1.0
            adjusted_demand = base_demand * weekday_multiplier
            
            timeline.append({
                'date': current_date,
                'days_to_event': days_to_event,
                'phase': phase_info['name'],
                'base_demand': base_demand,
                'adjusted_demand': adjusted_demand,
                'cumulative_demand': 0,  # Will calculate later
                'infrastructure_load': self._estimate_infrastructure_load(adjusted_demand, phase_info)
            })
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(timeline)
        
        # Calculate cumulative demand ensuring total matches tickets_sold
        df['cumulative_demand'] = df['adjusted_demand'].cumsum()
        total_demand = df['adjusted_demand'].sum()
        if total_demand > 0:
            scaling_factor = tickets_sold / total_demand
            df['adjusted_demand'] *= scaling_factor
            df['cumulative_demand'] = df['adjusted_demand'].cumsum()
            df['infrastructure_load'] *= scaling_factor
        
        return df
    
    def _determine_phase(self, current_date: date, start_date: date, event_date: date, phases: Dict) -> Dict:
        """Determine which demand phase we're in"""
        days_from_start = (current_date - start_date).days
        days_to_event = (event_date - current_date).days
        
        if days_to_event == 0:
            return {'name': 'event_day_peak', **phases['event_day_peak']}
        elif days_to_event <= phases['pre_event_surge']['duration_weeks'] * 7:
            return {'name': 'pre_event_surge', **phases['pre_event_surge']}
        elif days_from_start <= phases['launch_peak']['duration_weeks'] * 7:
            return {'name': 'launch_peak', **phases['launch_peak']}
        else:
            return {'name': 'steady_decline', **phases['steady_decline']}
    
    def _calculate_daily_demand(self, current_date: date, start_date: date, event_date: date, 
                               tickets_sold: int, phase_info: Dict) -> float:
        """Calculate base daily demand based on phase"""
        total_days = (event_date - start_date).days + 1
        average_daily = tickets_sold / total_days
        
        return average_daily * phase_info['intensity']
    
    def _estimate_infrastructure_load(self, demand: float, phase_info: Dict) -> float:
        """Estimate infrastructure load based on demand and phase type"""
        base_load = demand
        
        # Different phases stress infrastructure differently
        if phase_info['name'] == 'event_day_peak':
            # QR scanning is very infrastructure intensive
            base_load *= 3.0
        elif phase_info['name'] == 'launch_peak':
            # High concurrent users, payment processing stress
            base_load *= 2.0
        elif phase_info['name'] == 'pre_event_surge':
            # Moderate stress
            base_load *= 1.5
        else:
            # Steady state
            base_load *= 1.0
        
        return base_load

class TemporalInfrastructurePlanner:
    """Plans infrastructure across 2-week periods for optimal cost efficiency"""
    
    def __init__(self):
        self.infrastructure_tiers = {
            'Básico': {'aws_cost': 64.00, 'capacity_score': 1.0, 'description': 'Basic tier'},
            'Moderado': {'aws_cost': 245.00, 'capacity_score': 3.8, 'description': 'Moderate tier'},
            'Intenso': {'aws_cost': 735.00, 'capacity_score': 11.5, 'description': 'Intensive tier'},
            'Máximo': {'aws_cost': 3166.00, 'capacity_score': 49.5, 'description': 'Maximum tier'},
        }
        
        self.salary_costs = {
            'Básico': 900.00,
            'Moderado': 1200.00,
            'Intenso': 1800.00,
            'Máximo': 3000.00
        }
    
    def calculate_optimized_tier_cost(self, load_intensity: float) -> dict:
        """Calculate optimized infrastructure cost based on exact requirements"""
        # Base AWS costs for scaling calculation
        base_costs = {
            'rds_base': 15.40,      # db.t3.micro
            'ec2_base': 4.00,       # t3.nano
            'redis_base': 12.96,    # cache.t3.micro
            'additional': 25.00     # SES, WAF, etc.
        }
        
        # Scale factors based on load intensity
        if load_intensity <= 10:
            rds_multiplier = 1.0    # t3.micro
            ec2_multiplier = 1.0    # t3.nano
            redis_multiplier = 1.0  # t3.micro
        elif load_intensity <= 50:
            rds_multiplier = 2.5    # t3.small
            ec2_multiplier = 2.0    # t3.micro
            redis_multiplier = 1.5  # t3.small
        elif load_intensity <= 150:
            rds_multiplier = 8.0    # t3.medium
            ec2_multiplier = 4.0    # t3.small
            redis_multiplier = 3.0  # t3.medium
        elif load_intensity <= 400:
            rds_multiplier = 20.0   # t3.large
            ec2_multiplier = 10.0   # t3.medium
            redis_multiplier = 6.0  # t3.large
        else:
            rds_multiplier = 50.0   # t3.xlarge+
            ec2_multiplier = 25.0   # t3.large+
            redis_multiplier = 15.0 # r5.large+
        
        # Calculate optimized costs
        rds_cost = base_costs['rds_base'] * rds_multiplier
        ec2_cost = base_costs['ec2_base'] * ec2_multiplier
        redis_cost = base_costs['redis_base'] * redis_multiplier
        additional_cost = base_costs['additional']
        
        total_optimized_cost = rds_cost + ec2_cost + redis_cost + additional_cost
        
        return {
            'aws_cost': total_optimized_cost,
            'capacity_score': load_intensity / 10,  # Dynamic capacity score
            'description': f'Optimized for {load_intensity:.1f} load intensity',
            'breakdown': {
                'rds': rds_cost,
                'ec2': ec2_cost,
                'redis': redis_cost,
                'additional': additional_cost
            }
        }
    
    def create_campaign_timeline(self, events: List[Dict]) -> pd.DataFrame:
        """Create complete timeline with 2-week periods"""
        if not events:
            return pd.DataFrame()
        
        # Find campaign boundaries
        start_dates = [event.get('start_date', event['event_date']) for event in events]
        event_dates = [event['event_date'] for event in events]
        
        campaign_start = min(start_dates)
        campaign_end = max(event_dates)
        
        # Create 2-week periods
        periods = []
        current_start = campaign_start
        period_num = 1
        
        while current_start <= campaign_end:
            period_end = min(current_start + timedelta(days=13), campaign_end)  # 2 weeks = 14 days
            
            periods.append({
                'period_number': period_num,
                'start_date': current_start,
                'end_date': period_end,
                'duration_days': (period_end - current_start).days + 1
            })
            
            current_start = period_end + timedelta(days=1)
            period_num += 1
        
        return pd.DataFrame(periods)
    
    def calculate_period_demand(self, events: List[Dict], periods_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregated demand for each 2-week period"""
        demand_modeler = TemporalDemandModeler()
        period_demands = []
        
        for _, period in periods_df.iterrows():
            period_demand = 0
            period_load = 0
            active_events = 0
            event_details = []
            
            for event in events:
                # Check if event is active during this period
                event_start = event.get('start_date', event['event_date'])
                event_end = event['event_date']
                
                # Event is active if its timeline overlaps with period
                if event_start <= period['end_date'] and event_end >= period['start_date']:
                    active_events += 1
                    tickets_sold = int(event['num_tickets'] * event['occupancy_rate'] / 100)
                    
                    # Generate demand curve for this event
                    demand_curve = demand_modeler.calculate_demand_curve(
                        event_start, event_end, tickets_sold
                    )
                    
                    # Filter to period dates
                    period_curve = demand_curve[
                        (demand_curve['date'] >= period['start_date']) & 
                        (demand_curve['date'] <= period['end_date'])
                    ]
                    
                    if not period_curve.empty:
                        event_period_demand = period_curve['adjusted_demand'].sum()
                        event_period_load = period_curve['infrastructure_load'].sum()
                        
                        period_demand += event_period_demand
                        period_load += event_period_load
                        
                        event_details.append({
                            'name': event['name'],
                            'demand': event_period_demand,
                            'load': event_period_load
                        })
            
            period_demands.append({
                'period_number': period['period_number'],
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'duration_days': period['duration_days'],
                'total_demand': period_demand,
                'infrastructure_load': period_load,
                'active_events': active_events,
                'event_details': event_details,
                'load_intensity': period_load / period['duration_days'] if period['duration_days'] > 0 else 0
            })
        
        return pd.DataFrame(period_demands)
    
    def size_infrastructure_for_periods(self, period_demands_df: pd.DataFrame) -> pd.DataFrame:
        """Size infrastructure for each period based on load requirements"""
        sized_periods = []
        
        for _, period in period_demands_df.iterrows():
            load_intensity = period['load_intensity']
            
            # Calculate optimized tier first
            optimized_tier = self.calculate_optimized_tier_cost(load_intensity)
            
            # Determine predefined tier recommendations
            predefined_recommendations = []
            if load_intensity <= 50:
                predefined_recommendations.append('Básico')
            if load_intensity <= 200:
                predefined_recommendations.append('Moderado')
            if load_intensity <= 800:
                predefined_recommendations.append('Intenso')
            predefined_recommendations.append('Máximo')  # Always suitable
            
            # Find best option among predefined tiers
            best_predefined = None
            best_predefined_cost = float('inf')
            
            for tier_name in predefined_recommendations:
                tier_cost = self.infrastructure_tiers[tier_name]['aws_cost']
                if tier_cost < best_predefined_cost:
                    best_predefined = tier_name
                    best_predefined_cost = tier_cost
            
            # Compare optimized vs best predefined
            period_duration_weeks = period['duration_days'] / 7
            optimized_cost = optimized_tier['aws_cost'] * period_duration_weeks / 2
            predefined_cost = best_predefined_cost * period_duration_weeks / 2
            
            # Choose the better option
            if optimized_cost < predefined_cost * 0.85:  # 15% savings threshold
                recommended_tier = 'Optimizado'
                recommended_cost = optimized_cost
                tier_info = optimized_tier
            else:
                recommended_tier = best_predefined
                recommended_cost = predefined_cost
                tier_info = self.infrastructure_tiers[best_predefined]
            
            # Alternative tier analysis (include optimized + predefined tiers)
            alternatives = {}
            
            # Add optimized option
            alternatives['Optimizado'] = {
                'aws_cost': optimized_cost,
                'total_cost': optimized_cost,
                'suitable': True,  # Always suitable by design
                'description': optimized_tier['description'],
                'breakdown': optimized_tier['breakdown']
            }
            
            # Add predefined tiers
            for tier_name, tier_data in self.infrastructure_tiers.items():
                alt_aws = tier_data['aws_cost'] * period_duration_weeks / 2
                is_suitable = tier_data['capacity_score'] >= load_intensity / 100
                
                alternatives[tier_name] = {
                    'aws_cost': alt_aws,
                    'total_cost': alt_aws,
                    'suitable': is_suitable,
                    'description': tier_data['description']
                }
            
            sized_periods.append({
                'period_number': period['period_number'],
                'start_date': period['start_date'],
                'end_date': period['end_date'],
                'duration_days': period['duration_days'],
                'total_demand': period['total_demand'],
                'infrastructure_load': period['infrastructure_load'],
                'load_intensity': load_intensity,
                'active_events': period['active_events'],
                'event_details': period['event_details'],
                'recommended_tier': recommended_tier,
                'aws_cost': recommended_cost,
                'total_cost': recommended_cost,
                'alternatives': alternatives
            })
        
        return pd.DataFrame(sized_periods)
    
    def calculate_optimization_opportunities(self, sized_periods_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify cost optimization opportunities"""
        if sized_periods_df.empty:
            return {}
        
        total_optimized_cost = sized_periods_df['total_cost'].sum()
        
        # Calculate what it would cost with a single tier for entire campaign
        max_load = sized_periods_df['load_intensity'].max()
        
        if max_load <= 50:
            uniform_tier = 'Básico'
        elif max_load <= 200:
            uniform_tier = 'Moderado'
        elif max_load <= 800:
            uniform_tier = 'Intenso'
        else:
            uniform_tier = 'Máximo'
        
        # Calculate uniform tier cost (AWS only - salaries are flat monthly rate)
        total_weeks = sized_periods_df['duration_days'].sum() / 7
        uniform_aws_cost = self.infrastructure_tiers[uniform_tier]['aws_cost'] * total_weeks / 2
        uniform_total_cost = uniform_aws_cost  # Only AWS costs for temporal optimization
        
        savings = uniform_total_cost - total_optimized_cost
        savings_percentage = (savings / uniform_total_cost * 100) if uniform_total_cost > 0 else 0
        
        # Identify low-usage periods
        low_usage_periods = sized_periods_df[sized_periods_df['load_intensity'] < 30]
        high_usage_periods = sized_periods_df[sized_periods_df['load_intensity'] > 400]
        
        return {
            'total_optimized_cost': total_optimized_cost,
            'uniform_tier_cost': uniform_total_cost,
            'total_savings': savings,
            'savings_percentage': savings_percentage,
            'uniform_tier_needed': uniform_tier,
            'periods_count': len(sized_periods_df),
            'low_usage_periods': len(low_usage_periods),
            'high_usage_periods': len(high_usage_periods),
            'optimization_recommendations': self._generate_optimization_recommendations(sized_periods_df)
        }
    
    def _generate_optimization_recommendations(self, sized_periods_df: pd.DataFrame) -> List[str]:
        """Generate specific optimization recommendations"""
        recommendations = []
        
        # Check for consecutive low-usage periods
        consecutive_low = 0
        for _, period in sized_periods_df.iterrows():
            if period['load_intensity'] < 50:
                consecutive_low += 1
            else:
                if consecutive_low >= 3:
                    recommendations.append(
                        f"Consider scheduling maintenance during {consecutive_low} consecutive low-usage periods"
                    )
                consecutive_low = 0
        
        # Check for over-provisioning
        over_provisioned = sized_periods_df[
            sized_periods_df.apply(
                lambda row: any(
                    alt['suitable'] and alt['total_cost'] < row['total_cost'] * 0.8
                    for alt in row['alternatives'].values()
                ), axis=1
            )
        ]
        
        if len(over_provisioned) > 0:
            recommendations.append(
                f"Review {len(over_provisioned)} periods that may be over-provisioned"
            )
        
        # Check load distribution
        load_std = sized_periods_df['load_intensity'].std()
        load_mean = sized_periods_df['load_intensity'].mean()
        
        if load_std > load_mean * 0.5:
            recommendations.append(
                "High load variation detected - temporal scaling provides significant savings"
            )
        
        return recommendations

def show_temporal_infrastructure_analysis(events: List[Dict]):
    """Display temporal infrastructure planning interface"""
    
    if not events:
        st.warning("⚠️ Add events to see temporal infrastructure planning")
        return
    
    # Calculate campaign duration for title
    start_dates = [event.get('start_date', event['event_date']) for event in events]
    event_dates = [event['event_date'] for event in events]
    campaign_start = min(start_dates)
    campaign_end = max(event_dates)
    campaign_duration_days = (campaign_end - campaign_start).days + 1
    
    st.header("📊 Temporal Infrastructure Planning")
    st.markdown(f"**Right-size infrastructure across 2-week periods • Campaign Duration: {campaign_duration_days} days**")
    st.markdown("**🔧 AWS Infrastructure costs only - salaries are flat monthly rate regardless of tier**")
    
    # Initialize planner
    planner = TemporalInfrastructurePlanner()
    
    # Create timeline and analyze demand
    timeline_df = planner.create_campaign_timeline(events)
    period_demands_df = planner.calculate_period_demand(events, timeline_df)
    sized_periods_df = planner.size_infrastructure_for_periods(period_demands_df)
    optimization_analysis = planner.calculate_optimization_opportunities(sized_periods_df)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Planning Periods", len(sized_periods_df))
    with col2:
        st.metric("Total Cost", f"${optimization_analysis.get('total_optimized_cost', 0):,.2f}")
    with col3:
        st.metric("Potential Savings", f"${optimization_analysis.get('total_savings', 0):,.2f}")
    with col4:
        st.metric("Savings %", f"{optimization_analysis.get('savings_percentage', 0):.1f}%")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Demand Timeline", 
        "🏗️ Infrastructure Planning", 
        "💰 Cost Optimization", 
        "📊 Period Details"
    ])
    
    with tab1:
        st.subheader("Demand Patterns Over Time")
        
        # Show demand timeline visualization
        if not sized_periods_df.empty:
            fig = go.Figure()
            
            # Add demand bars
            fig.add_trace(go.Bar(
                x=sized_periods_df['period_number'],
                y=sized_periods_df['total_demand'],
                name='Ticket Demand',
                marker_color='lightblue'
            ))
            
            # Add infrastructure load line
            fig.add_trace(go.Scatter(
                x=sized_periods_df['period_number'],
                y=sized_periods_df['infrastructure_load'],
                mode='lines+markers',
                name='Infrastructure Load',
                yaxis='y2',
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title='Demand and Infrastructure Load by Period',
                xaxis_title='2-Week Period',
                yaxis_title='Ticket Demand',
                yaxis2=dict(
                    title='Infrastructure Load',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Show period summary table
        if not sized_periods_df.empty:
            display_df = sized_periods_df[['period_number', 'start_date', 'end_date', 
                                         'active_events', 'total_demand', 'load_intensity']].copy()
            display_df.columns = ['Period', 'Start Date', 'End Date', 'Active Events', 
                                'Demand', 'Load Intensity']
            display_df['Demand'] = display_df['Demand'].round(1)
            display_df['Load Intensity'] = display_df['Load Intensity'].round(1)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Infrastructure Tier Recommendations")
        
        if not sized_periods_df.empty:
            # Infrastructure tier timeline
            fig_tiers = go.Figure()
            
            tier_colors = {
                'Básico': 'green',
                'Moderado': 'orange', 
                'Intenso': 'red',
                'Máximo': 'purple',
                'Optimizado': 'blue'  # Add optimized tier color
            }
            
            for tier in tier_colors.keys():
                tier_periods = sized_periods_df[sized_periods_df['recommended_tier'] == tier]
                if not tier_periods.empty:
                    fig_tiers.add_trace(go.Bar(
                        x=tier_periods['period_number'],
                        y=[1] * len(tier_periods),  # Just for visualization
                        name=tier,
                        marker_color=tier_colors[tier],
                        text=[f"Period {p}" for p in tier_periods['period_number']],
                        textposition='inside'
                    ))
            
            fig_tiers.update_layout(
                title='Recommended Infrastructure Tiers by Period',
                xaxis_title='2-Week Period',
                yaxis_title='Infrastructure Tier',
                barmode='stack',
                showlegend=True
            )
            
            st.plotly_chart(fig_tiers, use_container_width=True)
            
            # Tier recommendations table
            tier_summary_df = sized_periods_df[['period_number', 'start_date', 'end_date',
                                              'recommended_tier', 'total_cost', 'load_intensity']].copy()
            tier_summary_df.columns = ['Period', 'Start Date', 'End Date', 'Tier', 
                                     'Cost', 'Load Intensity']
            tier_summary_df['Cost'] = tier_summary_df['Cost'].apply(lambda x: f"${x:,.2f}")
            tier_summary_df['Load Intensity'] = tier_summary_df['Load Intensity'].round(1)
            
            st.dataframe(tier_summary_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Cost Optimization Analysis")
        
        if optimization_analysis:
            # Savings comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**💡 Temporal Optimization Benefits (AWS Infrastructure Only):**")
                st.success(f"**AWS Infrastructure Savings: ${optimization_analysis['total_savings']:,.2f}**")
                st.info(f"**AWS Savings Percentage: {optimization_analysis['savings_percentage']:.1f}%**")
                st.write(f"**Optimized AWS Cost:** ${optimization_analysis['total_optimized_cost']:,.2f}")
                st.write(f"**Single Tier AWS Cost:** ${optimization_analysis['uniform_tier_cost']:,.2f}")
                st.write(f"**Single Tier Needed:** {optimization_analysis['uniform_tier_needed']}")
                st.caption("💡 Salaries remain constant monthly regardless of infrastructure tier")
            
            with col2:
                st.markdown("**📊 Period Analysis:**")
                st.metric("Total Periods", optimization_analysis['periods_count'])
                st.metric("Low Usage Periods", optimization_analysis['low_usage_periods'])
                st.metric("High Usage Periods", optimization_analysis['high_usage_periods'])
            
            # Recommendations
            if optimization_analysis['optimization_recommendations']:
                st.markdown("**🎯 Optimization Recommendations:**")
                for rec in optimization_analysis['optimization_recommendations']:
                    st.write(f"• {rec}")
            
            # Cost comparison chart
            if not sized_periods_df.empty:
                fig_costs = go.Figure()
                
                fig_costs.add_trace(go.Bar(
                    x=sized_periods_df['period_number'],
                    y=sized_periods_df['total_cost'],
                    name='Optimized Cost',
                    marker_color='green'
                ))
                
                # Add uniform tier cost line for comparison (AWS only)
                uniform_tier = optimization_analysis['uniform_tier_needed']
                uniform_period_cost = planner.infrastructure_tiers[uniform_tier]['aws_cost'] / 2  # 2-week period (AWS only)
                
                fig_costs.add_trace(go.Scatter(
                    x=sized_periods_df['period_number'],
                    y=[uniform_period_cost] * len(sized_periods_df),
                    mode='lines',
                    name=f'Uniform {uniform_tier} Tier',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_costs.update_layout(
                    title='AWS Infrastructure Cost Comparison: Temporal Optimization vs Single Tier',
                    xaxis_title='2-Week Period',
                    yaxis_title='AWS Infrastructure Cost (USD)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_costs, use_container_width=True)
    
    with tab4:
        st.subheader("Detailed Period Analysis")
        
        if not sized_periods_df.empty:
            # Period selector
            selected_period = st.selectbox(
                "Select Period for Details",
                sized_periods_df['period_number'].tolist(),
                format_func=lambda x: f"Period {x}"
            )
            
            period_data = sized_periods_df[sized_periods_df['period_number'] == selected_period].iloc[0]
            
            # Period details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**📅 Period {period_data['period_number']} Details**")
                st.write(f"**Duration:** {period_data['start_date']} to {period_data['end_date']}")
                st.write(f"**Days:** {period_data['duration_days']}")
                st.write(f"**Active Events:** {period_data['active_events']}")
                st.write(f"**Total Demand:** {period_data['total_demand']:.1f} tickets")
                st.write(f"**Load Intensity:** {period_data['load_intensity']:.1f}")
                st.write(f"**Recommended Tier:** {period_data['recommended_tier']}")
                st.write(f"**Total Cost:** ${period_data['total_cost']:,.2f}")
            
            with col2:
                st.markdown("**🔄 Alternative Tier Options:**")
                alternatives = period_data['alternatives']
                alt_data = []
                
                for tier_name, alt_info in alternatives.items():
                    suitable_icon = "✅" if alt_info['suitable'] else "❌"
                    
                    # Format cost display
                    cost_display = f"${alt_info['total_cost']:,.2f}"
                    
                    # Add breakdown for optimized tier
                    if tier_name == 'Optimizado' and 'breakdown' in alt_info:
                        breakdown = alt_info['breakdown']
                        cost_display += f" (RDS: ${breakdown['rds']:.0f}, EC2: ${breakdown['ec2']:.0f}, Redis: ${breakdown['redis']:.0f})"
                    
                    alt_data.append({
                        'Tier': tier_name,  # Remove green checkmarks from tier names
                        'Total Cost': cost_display,
                        'Suitable': suitable_icon,  # Only show checkmarks here
                        'Description': alt_info.get('description', '')
                    })
                
                alt_df = pd.DataFrame(alt_data)
                st.dataframe(alt_df, use_container_width=True, hide_index=True)
            
            # Event details for this period
            if period_data['event_details']:
                st.markdown(f"**🎭 Events Active in Period {period_data['period_number']}:**")
                event_details_df = pd.DataFrame(period_data['event_details'])
                event_details_df['demand'] = event_details_df['demand'].round(1)
                event_details_df['load'] = event_details_df['load'].round(1)
                event_details_df.columns = ['Event Name', 'Demand', 'Infrastructure Load']
                st.dataframe(event_details_df, use_container_width=True, hide_index=True) 