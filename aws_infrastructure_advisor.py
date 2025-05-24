import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import streamlit as st

class AWSInfrastructureAdvisor:
    """
    Smart infrastructure advisor that analyzes event campaigns and recommends 
    optimal AWS tier configurations based on real AWS pricing and performance data.
    """
    
    def __init__(self):
        # AWS RDS Instance Performance Data (connections & storage capacity)
        self.rds_instances = {
            'db.t3.micro': {
                'vcpu': 2, 'memory_gb': 1, 'max_connections': 85, 
                'cost_monthly': 15, 'performance_score': 1
            },
            'db.t3.small': {
                'vcpu': 2, 'memory_gb': 2, 'max_connections': 170,
                'cost_monthly': 30, 'performance_score': 2
            },
            'db.r5.large': {
                'vcpu': 2, 'memory_gb': 16, 'max_connections': 1000,
                'cost_monthly': 74, 'performance_score': 4
            },
            'db.r5.xlarge': {
                'vcpu': 4, 'memory_gb': 32, 'max_connections': 2000,
                'cost_monthly': 148, 'performance_score': 6
            },
            'db.r5.2xlarge': {
                'vcpu': 8, 'memory_gb': 64, 'max_connections': 3000,
                'cost_monthly': 296, 'performance_score': 8
            }
        }
        
        # AWS EC2 Instance Performance Data (concurrent users handling)
        self.ec2_instances = {
            't3.micro': {
                'vcpu': 2, 'memory_gb': 1, 'concurrent_users': 50,
                'cost_monthly': 8.5, 'performance_score': 1
            },
            't3.small': {
                'vcpu': 2, 'memory_gb': 2, 'concurrent_users': 100,
                'cost_monthly': 17, 'performance_score': 2
            },
            't3.medium': {
                'vcpu': 2, 'memory_gb': 4, 'concurrent_users': 200,
                'cost_monthly': 34, 'performance_score': 3
            },
            'c5.large': {
                'vcpu': 2, 'memory_gb': 4, 'concurrent_users': 300,
                'cost_monthly': 62, 'performance_score': 4
            },
            'c5.xlarge': {
                'vcpu': 4, 'memory_gb': 8, 'concurrent_users': 600,
                'cost_monthly': 124, 'performance_score': 6
            },
            'c5.2xlarge': {
                'vcpu': 8, 'memory_gb': 16, 'concurrent_users': 1200,
                'cost_monthly': 248, 'performance_score': 8
            }
        }
        
        # Infrastructure Tier Definitions (from user's original data)
        self.infrastructure_tiers = {
            'Básico': {
                'total_cost': 964,
                'salary_cost': 900,
                'aws_cost': 64,  # 964 - 900
                'max_events': 5,
                'max_concurrent_users': 500,
                'max_db_transactions': 1000,
                'description': 'Small campaigns, low concurrent usage'
            },
            'Moderado': {
                'total_cost': 1445,
                'salary_cost': 1200,
                'aws_cost': 245,  # 1445 - 1200
                'max_events': 15,
                'max_concurrent_users': 2000,
                'max_db_transactions': 5000,
                'description': 'Medium campaigns, moderate concurrent usage'
            },
            'Intenso': {
                'total_cost': 2535,
                'salary_cost': 1800,
                'aws_cost': 735,  # 2535 - 1800
                'max_events': 30,
                'max_concurrent_users': 5000,
                'max_db_transactions': 15000,
                'description': 'Large campaigns, high concurrent usage'
            },
            'Máximo': {
                'total_cost': 6166,
                'salary_cost': 3000,
                'aws_cost': 3166,  # 6166 - 3000
                'max_events': 100,
                'max_concurrent_users': 15000,
                'max_db_transactions': 50000,
                'description': 'Enterprise campaigns, very high concurrent usage'
            }
        }
    
    def analyze_campaign_requirements(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze campaign events and calculate infrastructure requirements."""
        if not events:
            return {}
        
        total_events = len(events)
        total_tickets = sum(event['num_tickets'] for event in events)
        total_sold_tickets = sum(int(event['num_tickets'] * event['occupancy_rate'] / 100) for event in events)
        
        # Calculate peak concurrent usage (assume 20% of sold tickets are concurrent)
        peak_concurrent_users = int(total_sold_tickets * 0.20)
        
        # Estimate database transactions per hour during peak (tickets + browsing)
        # Assume each concurrent user generates 10 transactions/hour
        peak_db_transactions = peak_concurrent_users * 10
        
        # Calculate campaign duration
        event_dates = [event['event_date'] for event in events]
        campaign_start = min(event_dates)
        campaign_end = max(event_dates)
        campaign_duration_days = (campaign_end - campaign_start).days + 1
        
        # Estimate storage requirements (GB)
        # Base: 1GB + 0.1GB per ticket + 0.5GB per event
        estimated_storage_gb = 1 + (total_tickets * 0.0001) + (total_events * 0.5)
        
        return {
            'total_events': total_events,
            'total_tickets': total_tickets,
            'total_sold_tickets': total_sold_tickets,
            'peak_concurrent_users': peak_concurrent_users,
            'peak_db_transactions': peak_db_transactions,
            'campaign_duration_days': campaign_duration_days,
            'estimated_storage_gb': estimated_storage_gb,
            'campaign_start': campaign_start,
            'campaign_end': campaign_end
        }
    
    def recommend_rds_instance(self, requirements: Dict[str, Any]) -> Tuple[str, Dict]:
        """Recommend optimal RDS instance based on requirements."""
        peak_transactions = requirements.get('peak_db_transactions', 0)
        storage_gb = requirements.get('estimated_storage_gb', 1)
        
        # Find minimum instance that can handle the load
        for instance_name, specs in self.rds_instances.items():
            # Check if instance can handle the transaction load
            # Assume max_connections correlates with transaction capacity
            if specs['max_connections'] >= peak_transactions:
                return instance_name, specs
        
        # If no instance is sufficient, return the largest
        largest_instance = max(self.rds_instances.items(), 
                             key=lambda x: x[1]['max_connections'])
        return largest_instance[0], largest_instance[1]
    
    def recommend_ec2_instance(self, requirements: Dict[str, Any]) -> Tuple[str, Dict]:
        """Recommend optimal EC2 instance based on requirements."""
        peak_users = requirements.get('peak_concurrent_users', 0)
        
        # Find minimum instance that can handle concurrent users
        for instance_name, specs in self.ec2_instances.items():
            if specs['concurrent_users'] >= peak_users:
                return instance_name, specs
        
        # If no instance is sufficient, return the largest
        largest_instance = max(self.ec2_instances.items(), 
                             key=lambda x: x[1]['concurrent_users'])
        return largest_instance[0], largest_instance[1]
    
    def recommend_infrastructure_tier(self, events: List[Dict]) -> Dict[str, Any]:
        """Recommend optimal infrastructure tier based on campaign analysis."""
        requirements = self.analyze_campaign_requirements(events)
        
        if not requirements:
            return {
                'recommended_tier': 'Básico',
                'tier_info': self.infrastructure_tiers['Básico'],
                'requirements': {},
                'analysis': {},
                'confidence': 'Low - No events provided'
            }
        
        # Get component recommendations
        rds_instance, rds_specs = self.recommend_rds_instance(requirements)
        ec2_instance, ec2_specs = self.recommend_ec2_instance(requirements)
        
        # Calculate estimated AWS costs
        estimated_rds_cost = rds_specs['cost_monthly']
        estimated_ec2_cost = ec2_specs['cost_monthly']
        estimated_storage_cost = max(20, requirements['estimated_storage_gb'] * 0.115)  # GP2 pricing
        estimated_backup_cost = estimated_storage_cost * 0.2  # 20% of storage for backups
        estimated_data_transfer = 10  # Base estimate for data transfer
        
        total_estimated_aws_cost = (
            estimated_rds_cost + 
            estimated_ec2_cost + 
            estimated_storage_cost + 
            estimated_backup_cost + 
            estimated_data_transfer
        )
        
        # Find best fitting tier
        recommended_tier = 'Básico'
        confidence_score = 0
        
        for tier_name, tier_specs in self.infrastructure_tiers.items():
            fits_events = requirements['total_events'] <= tier_specs['max_events']
            fits_users = requirements['peak_concurrent_users'] <= tier_specs['max_concurrent_users']
            fits_transactions = requirements['peak_db_transactions'] <= tier_specs['max_db_transactions']
            fits_budget = total_estimated_aws_cost <= tier_specs['aws_cost'] * 1.1  # 10% tolerance
            
            if fits_events and fits_users and fits_transactions:
                recommended_tier = tier_name
                if fits_budget:
                    confidence_score = 95
                    break
                else:
                    confidence_score = 75
        
        # Determine confidence level
        if confidence_score >= 90:
            confidence_level = 'High'
        elif confidence_score >= 70:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'recommended_tier': recommended_tier,
            'tier_info': self.infrastructure_tiers[recommended_tier],
            'requirements': requirements,
            'analysis': {
                'recommended_rds': rds_instance,
                'rds_specs': rds_specs,
                'recommended_ec2': ec2_instance,
                'ec2_specs': ec2_specs,
                'estimated_costs': {
                    'rds_monthly': estimated_rds_cost,
                    'ec2_monthly': estimated_ec2_cost,
                    'storage_monthly': estimated_storage_cost,
                    'backup_monthly': estimated_backup_cost,
                    'data_transfer_monthly': estimated_data_transfer,
                    'total_aws_monthly': total_estimated_aws_cost
                },
                'performance_requirements': {
                    'peak_concurrent_users': requirements['peak_concurrent_users'],
                    'peak_db_transactions': requirements['peak_db_transactions'],
                    'storage_needs_gb': requirements['estimated_storage_gb']
                }
            },
            'confidence': f'{confidence_level} ({confidence_score}%)',
            'recommendations': self._generate_recommendations(requirements, recommended_tier)
        }
    
    def _generate_recommendations(self, requirements: Dict, recommended_tier: str) -> List[str]:
        """Generate specific recommendations for optimization."""
        recommendations = []
        
        peak_users = requirements.get('peak_concurrent_users', 0)
        total_events = requirements.get('total_events', 0)
        
        if peak_users < 100:
            recommendations.append("Consider using AWS RDS Proxy to optimize database connections")
        
        if total_events > 20:
            recommendations.append("Implement database read replicas for better performance")
            recommendations.append("Consider using Amazon CloudFront CDN for static content")
        
        if peak_users > 1000:
            recommendations.append("Use Auto Scaling Groups for EC2 instances")
            recommendations.append("Implement Application Load Balancer for traffic distribution")
        
        if recommended_tier in ['Intenso', 'Máximo']:
            recommendations.append("Consider Multi-AZ deployment for high availability")
            recommendations.append("Implement comprehensive monitoring with CloudWatch")
        
        return recommendations
    
    def generate_cost_comparison(self, events: List[Dict]) -> Dict[str, Any]:
        """Generate cost comparison across all tiers."""
        recommendation = self.recommend_infrastructure_tier(events)
        
        comparison = {}
        for tier_name, tier_specs in self.infrastructure_tiers.items():
            is_recommended = tier_name == recommendation['recommended_tier']
            fits_requirements = self._check_tier_fitness(
                recommendation['requirements'], tier_specs
            )
            
            comparison[tier_name] = {
                'aws_cost_monthly': tier_specs['aws_cost'],
                'total_cost_monthly': tier_specs['total_cost'],
                'fits_requirements': fits_requirements,
                'is_recommended': is_recommended,
                'description': tier_specs['description'],
                'capacity': {
                    'max_events': tier_specs['max_events'],
                    'max_concurrent_users': tier_specs['max_concurrent_users'],
                    'max_db_transactions': tier_specs['max_db_transactions']
                }
            }
        
        return comparison
    
    def _check_tier_fitness(self, requirements: Dict, tier_specs: Dict) -> Dict[str, bool]:
        """Check if a tier can handle the requirements."""
        if not requirements:
            return {'events': True, 'users': True, 'transactions': True}
        
        return {
            'events': requirements.get('total_events', 0) <= tier_specs['max_events'],
            'users': requirements.get('peak_concurrent_users', 0) <= tier_specs['max_concurrent_users'],
            'transactions': requirements.get('peak_db_transactions', 0) <= tier_specs['max_db_transactions']
        } 