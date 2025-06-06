import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
import pandas as pd

class EnhancedAWSInfrastructureAdvisor:
    """
    Enhanced AWS Infrastructure Advisor with granular operational analysis,
    custom tier generation, latency optimization, and comprehensive service coverage.
    """
    
    def __init__(self):
        # Detailed operation patterns for e-ticketing system
        self.operation_patterns = {
            'ticket_purchase': {
                'description': 'User purchases tickets',
                'operations': ['payment_processing', 'matic_minting', 'db_write', 'email_queue'],
                'peak_multiplier': 5.0,  # 5x higher during sale opening
                'duration_pattern': 'burst',  # High traffic in short bursts
                'latency_sensitive': True
            },
            'qr_generation': {
                'description': 'Generate QR codes and send emails',
                'operations': ['qr_generation', 'ses_email', 'storage_write'],
                'peak_multiplier': 1.0,  # Consistent load
                'duration_pattern': 'batch',  # Can be processed in batches
                'latency_sensitive': False
            },
            'event_day_scanning': {
                'description': 'Scan tickets at venue',
                'operations': ['qr_validation', 'db_read', 'cache_read'],
                'peak_multiplier': 10.0,  # Very high during event start
                'duration_pattern': 'spike',  # Concentrated in 2-3 hours
                'latency_sensitive': True
            },
            'browsing_tickets': {
                'description': 'Users browsing available tickets',
                'operations': ['db_read', 'cache_read', 'cdn_delivery'],
                'peak_multiplier': 3.0,  # Higher during presale announcements
                'duration_pattern': 'sustained',  # Longer periods
                'latency_sensitive': True
            }
        }
        
        # Comprehensive AWS service catalog with real pricing
        self.aws_services = {
            'rds_postgres': {
                'db.t3.micro': {'vcpu': 2, 'memory_gb': 1, 'max_connections': 85, 'cost_monthly': 15.40, 'storage_iops': 100},
                'db.t3.small': {'vcpu': 2, 'memory_gb': 2, 'max_connections': 170, 'cost_monthly': 30.80, 'storage_iops': 200},
                'db.t3.medium': {'vcpu': 2, 'memory_gb': 4, 'max_connections': 340, 'cost_monthly': 61.60, 'storage_iops': 400},
                'db.r5.large': {'vcpu': 2, 'memory_gb': 16, 'max_connections': 1000, 'cost_monthly': 148.00, 'storage_iops': 1000},
                'db.r5.xlarge': {'vcpu': 4, 'memory_gb': 32, 'max_connections': 2000, 'cost_monthly': 296.00, 'storage_iops': 2000},
                'db.r5.2xlarge': {'vcpu': 8, 'memory_gb': 64, 'max_connections': 3000, 'cost_monthly': 592.00, 'storage_iops': 3000}
            },
            'ec2_compute': {
                't3.nano': {'vcpu': 2, 'memory_gb': 0.5, 'concurrent_users': 25, 'cost_monthly': 4.00, 'network_performance': 'Low'},
                't3.micro': {'vcpu': 2, 'memory_gb': 1, 'concurrent_users': 50, 'cost_monthly': 8.50, 'network_performance': 'Low'},
                't3.small': {'vcpu': 2, 'memory_gb': 2, 'concurrent_users': 100, 'cost_monthly': 17.00, 'network_performance': 'Low to Moderate'},
                't3.medium': {'vcpu': 2, 'memory_gb': 4, 'concurrent_users': 200, 'cost_monthly': 34.00, 'network_performance': 'Low to Moderate'},
                'c5.large': {'vcpu': 2, 'memory_gb': 4, 'concurrent_users': 300, 'cost_monthly': 62.00, 'network_performance': 'Up to 10 Gbps'},
                'c5.xlarge': {'vcpu': 4, 'memory_gb': 8, 'concurrent_users': 600, 'cost_monthly': 124.00, 'network_performance': 'Up to 10 Gbps'},
                'c5.2xlarge': {'vcpu': 8, 'memory_gb': 16, 'concurrent_users': 1200, 'cost_monthly': 248.00, 'network_performance': 'Up to 10 Gbps'},
                'c5.4xlarge': {'vcpu': 16, 'memory_gb': 32, 'concurrent_users': 2400, 'cost_monthly': 496.00, 'network_performance': '10 Gbps'}
            },
            'elasticache_redis': {
                'cache.t3.micro': {'vcpu': 2, 'memory_gb': 0.5, 'cost_monthly': 11.00, 'max_connections': 65000},
                'cache.t3.small': {'vcpu': 2, 'memory_gb': 1.37, 'cost_monthly': 22.00, 'max_connections': 65000},
                'cache.r5.large': {'vcpu': 2, 'memory_gb': 13.07, 'cost_monthly': 126.00, 'max_connections': 65000},
                'cache.r5.xlarge': {'vcpu': 4, 'memory_gb': 26.32, 'cost_monthly': 252.00, 'max_connections': 65000}
            }
        }
        
        # Self-hosted Redis hardware options (Puebla, Mexico context)
        self.self_hosted_redis = {
            'mini_pc_basic': {
                'name': 'Intel NUC / Mini PC (Basic)',
                'specs': 'Intel i3, 8GB RAM, 256GB SSD',
                'hardware_cost_usd': 400,
                'monthly_electricity_kwh': 15,  # Very low power consumption
                'electricity_cost_per_kwh_usd': 0.08,  # Puebla, Mexico average
                'monthly_electricity_cost': 1.20,
                'depreciation_months': 36,  # 3 years
                'monthly_depreciation': 11.11,
                'monthly_total_cost': 12.31,
                'performance_equivalent': 'cache.t3.small',
                'memory_gb': 8,
                'suitable_for': 'Up to 10,000 tickets'
            },
            'mini_pc_performance': {
                'name': 'Xeon Mini PC (Performance)',
                'specs': 'Intel Xeon E3, 32GB RAM, 1TB SSD',
                'hardware_cost_usd': 800,
                'monthly_electricity_kwh': 25,
                'electricity_cost_per_kwh_usd': 0.08,
                'monthly_electricity_cost': 2.00,
                'depreciation_months': 36,
                'monthly_depreciation': 22.22,
                'monthly_total_cost': 24.22,
                'performance_equivalent': 'cache.r5.large',
                'memory_gb': 32,
                'suitable_for': 'Up to 50,000 tickets'
            },
            'mini_pc_enterprise': {
                'name': 'High-End Mini Server',
                'specs': 'Intel Xeon, 64GB RAM, 2TB NVMe',
                'hardware_cost_usd': 1500,
                'monthly_electricity_kwh': 35,
                'electricity_cost_per_kwh_usd': 0.08,
                'monthly_electricity_cost': 2.80,
                'depreciation_months': 36,
                'monthly_depreciation': 41.67,
                'monthly_total_cost': 44.47,
                'performance_equivalent': 'cache.r5.xlarge',
                'memory_gb': 64,
                'suitable_for': '100,000+ tickets'
            }
        }
        
        # Additional AWS service costs (monthly)
        self.additional_services = {
            'ses_email': {
                'description': 'Email sending via SES',
                'cost_per_1000_emails': 0.10,
                'base_monthly': 0.00
            },
            'waf': {
                'description': 'Web Application Firewall',
                'cost_per_web_acl': 1.00,
                'cost_per_rule': 2.00,
                'cost_per_million_requests': 0.60,
                'typical_monthly': 25.00  # For small-medium traffic
            },
            's3_storage': {
                'description': 'S3 storage for QR codes, images',
                'cost_per_gb_month': 0.023,
                'cost_per_1000_requests': 0.0004
            },
            'cloudfront_cdn': {
                'description': 'CDN for static assets',
                'cost_per_gb_transfer': 0.085,
                'cost_per_10000_requests': 0.0075,
                'typical_monthly': 15.00  # For small-medium traffic
            },
            'route53_dns': {
                'description': 'DNS management',
                'cost_per_hosted_zone': 0.50,
                'cost_per_million_queries': 0.40,
                'typical_monthly': 5.00
            },
            'lambda_functions': {
                'description': 'Serverless functions for QR generation, etc.',
                'cost_per_million_requests': 0.20,
                'cost_per_gb_second': 0.0000166667,
                'typical_monthly': 10.00  # For moderate usage
            },
            'netlify_hosting': {
                'description': 'Frontend hosting (external service)',
                'starter_plan': 0.00,
                'pro_plan': 19.00,
                'business_plan': 99.00
            }
        }
        
        # Predefined tiers (backward compatibility)
        self.predefined_tiers = {
            'Básico': {
                'max_events': 5,
                'max_concurrent_users': 500,
                'max_tickets_per_event': 1000,
                'max_db_transactions_hour': 1000,
                'target_latency_ms': 2000,
                'salary_cost': 900.00,
                'description': 'Small campaigns with basic infrastructure needs'
            },
            'Moderado': {
                'max_events': 15,
                'max_concurrent_users': 2000,
                'max_tickets_per_event': 5000,
                'max_db_transactions_hour': 5000,
                'target_latency_ms': 1000,
                'salary_cost': 1200.00,
                'description': 'Medium campaigns with moderate traffic and performance requirements'
            },
            'Intenso': {
                'max_events': 30,
                'max_concurrent_users': 5000,
                'max_tickets_per_event': 10000,
                'max_db_transactions_hour': 15000,
                'target_latency_ms': 500,
                'salary_cost': 1800.00,
                'description': 'Large campaigns requiring high performance and scalability'
            },
            'Máximo': {
                'max_events': 100,
                'max_concurrent_users': 15000,
                'max_tickets_per_event': 50000,
                'max_db_transactions_hour': 50000,
                'target_latency_ms': 200,
                'salary_cost': 3000.00,
                'description': 'Enterprise-level campaigns with maximum performance requirements'
            }
        }
    
    def analyze_operational_requirements(self, events: List[Dict]) -> Dict[str, Any]:
        """Deeply analyze operational patterns for each phase of the ticketing lifecycle."""
        if not events:
            return {}
        
        total_tickets = sum(event['num_tickets'] for event in events)
        total_sold_tickets = sum(int(event['num_tickets'] * event['occupancy_rate'] / 100) for event in events)
        
        # Operational analysis per phase
        operations = {
            'ticket_purchase_phase': {
                'peak_concurrent_buyers': self._calculate_peak_buyers(events),
                'db_writes_per_hour': self._calculate_db_writes(events),
                'payment_transactions_hour': self._calculate_payment_load(events),
                'matic_minting_operations': total_sold_tickets,
                'storage_writes_gb': total_sold_tickets * 0.0001  # Small metadata per ticket
            },
            'qr_email_phase': {
                'qr_generations_needed': total_sold_tickets,
                'emails_to_send': total_sold_tickets,
                'ses_email_cost': total_sold_tickets * 0.0001,  # $0.10 per 1000 emails
                'batch_processing_hours': max(1, total_sold_tickets / 1000),  # 1000 QRs per hour
                'storage_for_qrs_gb': total_sold_tickets * 0.002  # 2KB per QR image
            },
            'event_day_scanning': {
                'peak_scans_per_hour': self._calculate_peak_scanning(events),
                'db_reads_per_hour': self._calculate_scanning_reads(events),
                'cache_hits_required': self._calculate_cache_needs(events),
                'latency_requirement_ms': 100,  # Critical for user experience
                'concurrent_scanners': self._calculate_concurrent_scanners(events)
            },
            'browsing_phase': {
                'peak_browsers': self._calculate_peak_browsers(events),
                'cdn_requests_hour': self._calculate_cdn_load(events),
                'db_reads_per_hour': self._calculate_browsing_reads(events),
                'cache_usage_gb': max(1, total_tickets / 10000)  # Cache seat maps, etc.
            }
        }
        
        # Calculate infrastructure requirements based on operations
        requirements = {
            'peak_concurrent_users': max(
                operations['ticket_purchase_phase']['peak_concurrent_buyers'],
                operations['event_day_scanning']['concurrent_scanners'],
                operations['browsing_phase']['peak_browsers']
            ),
            'peak_db_connections': max(
                operations['ticket_purchase_phase']['db_writes_per_hour'] / 100,  # Writes need more connections
                operations['event_day_scanning']['db_reads_per_hour'] / 200,
                operations['browsing_phase']['db_reads_per_hour'] / 300
            ),
            'storage_requirements_gb': (
                operations['qr_email_phase']['storage_for_qrs_gb'] +
                operations['ticket_purchase_phase']['storage_writes_gb'] +
                5  # Base system storage
            ),
            'email_volume': operations['qr_email_phase']['emails_to_send'],
            'cache_memory_gb': max(1, operations['browsing_phase']['cache_usage_gb']),
            'latency_target_ms': min(
                operations['event_day_scanning']['latency_requirement_ms'],
                500  # General target
            ),
            'total_events': len(events),
            'total_tickets': total_tickets,
            'total_sold_tickets': total_sold_tickets
        }
        
        return {
            'operations': operations,
            'requirements': requirements,
            'cost_drivers': self._identify_cost_drivers(operations)
        }
    
    def _calculate_peak_buyers(self, events: List[Dict]) -> int:
        """Calculate peak concurrent buyers during ticket sales."""
        # Assume 10% of total tickets are bought in first hour, 20% of those are concurrent
        max_event_tickets = max(event['num_tickets'] for event in events)
        return int(max_event_tickets * 0.10 * 0.20)
    
    def _calculate_db_writes(self, events: List[Dict]) -> int:
        """Calculate database writes per hour during peak purchase time."""
        peak_buyers = self._calculate_peak_buyers(events)
        return peak_buyers * 5  # Each purchase = 5 DB operations
    
    def _calculate_payment_load(self, events: List[Dict]) -> int:
        """Calculate payment processing load."""
        return self._calculate_peak_buyers(events)  # 1:1 with concurrent buyers
    
    def _calculate_peak_scanning(self, events: List[Dict]) -> int:
        """Calculate peak ticket scanning rate on event day."""
        # Assume 80% of tickets scanned in 2-hour window
        max_sold_tickets = max(int(event['num_tickets'] * event['occupancy_rate'] / 100) for event in events)
        return int(max_sold_tickets * 0.80 / 2)  # Per hour
    
    def _calculate_scanning_reads(self, events: List[Dict]) -> int:
        """Calculate DB reads during scanning."""
        return self._calculate_peak_scanning(events) * 2  # 2 reads per scan (validate + update)
    
    def _calculate_cache_needs(self, events: List[Dict]) -> int:
        """Calculate cache requirements for fast ticket validation."""
        total_tickets = sum(event['num_tickets'] for event in events)
        return int(total_tickets * 0.5)  # Cache 50% of tickets for fast access
    
    def _calculate_concurrent_scanners(self, events: List[Dict]) -> int:
        """Calculate concurrent scanning devices/staff."""
        max_event_capacity = max(event['num_tickets'] for event in events)
        return max(5, int(max_event_capacity / 500))  # 1 scanner per 500 capacity
    
    def _calculate_peak_browsers(self, events: List[Dict]) -> int:
        """Calculate peak concurrent browsers."""
        max_event_tickets = max(event['num_tickets'] for event in events)
        return int(max_event_tickets * 0.05)  # 5% of capacity browsing concurrently
    
    def _calculate_cdn_load(self, events: List[Dict]) -> int:
        """Calculate CDN requests per hour."""
        peak_browsers = self._calculate_peak_browsers(events)
        return peak_browsers * 20  # 20 asset requests per browser per hour
    
    def _calculate_browsing_reads(self, events: List[Dict]) -> int:
        """Calculate DB reads from browsing activity."""
        peak_browsers = self._calculate_peak_browsers(events)
        return peak_browsers * 5  # 5 reads per browser per hour
    
    def _identify_cost_drivers(self, operations: Dict) -> List[str]:
        """Identify the main cost drivers based on operational analysis."""
        drivers = []
        
        # Check email volume
        if operations['qr_email_phase']['emails_to_send'] > 10000:
            drivers.append("High email volume - consider batch optimization")
        
        # Check scanning requirements
        if operations['event_day_scanning']['peak_scans_per_hour'] > 1000:
            drivers.append("High scanning load - requires Redis caching")
        
        # Check concurrent users
        if operations['ticket_purchase_phase']['peak_concurrent_buyers'] > 500:
            drivers.append("High concurrent purchase load - requires load balancing")
        
        return drivers
    
    def generate_custom_infrastructure(self, analysis: Dict) -> Dict[str, Any]:
        """Generate custom infrastructure recommendation when predefined tiers don't fit."""
        requirements = analysis['requirements']
        
        # Find optimal RDS instance
        rds_instance, rds_cost = self._optimize_rds_selection(requirements)
        
        # Find optimal EC2 instance  
        ec2_instance, ec2_cost = self._optimize_ec2_selection(requirements)
        
        # Find optimal Redis cache
        redis_instance, redis_cost = self._optimize_redis_selection(requirements)
        
        # Calculate additional service costs
        additional_costs = self._calculate_additional_service_costs(analysis)
        
        # Calculate total monthly cost
        total_aws_cost = (
            rds_cost + 
            ec2_cost + 
            redis_cost + 
            additional_costs['total']
        )
        
        # Determine appropriate salary tier based on complexity
        salary_tier = self._determine_salary_tier(requirements, total_aws_cost)
        
        custom_config = {
            'tier_name': 'Custom',
            'aws_components': {
                'rds': {'instance': rds_instance, 'cost': rds_cost},
                'ec2': {'instance': ec2_instance, 'cost': ec2_cost},
                'redis': {'instance': redis_instance, 'cost': redis_cost},
                'additional_services': additional_costs
            },
            'total_aws_cost': total_aws_cost,
            'salary_cost': salary_tier['cost'],
            'total_monthly_cost': total_aws_cost + salary_tier['cost'],
            'description': f"Custom tier optimized for {requirements['total_events']} events with {requirements['peak_concurrent_users']:,} peak users",
            'performance_profile': {
                'max_events': requirements['total_events'],
                'max_concurrent_users': requirements['peak_concurrent_users'],
                'latency_target_ms': requirements['latency_target_ms'],
                'storage_gb': requirements['storage_requirements_gb']
            },
            'optimization_notes': self._generate_optimization_notes(analysis)
        }
        
        return custom_config
    
    def _optimize_rds_selection(self, requirements: Dict) -> Tuple[str, float]:
        """Select optimal RDS instance based on connection and performance needs."""
        target_connections = requirements['peak_db_connections']
        
        for instance_name, specs in self.aws_services['rds_postgres'].items():
            if specs['max_connections'] >= target_connections * 1.2:  # 20% buffer
                return instance_name, specs['cost_monthly']
        
        # If no instance fits, return the largest
        largest = max(self.aws_services['rds_postgres'].items(), 
                     key=lambda x: x[1]['max_connections'])
        return largest[0], largest[1]['cost_monthly']
    
    def _optimize_ec2_selection(self, requirements: Dict) -> Tuple[str, float]:
        """Select optimal EC2 instance based on concurrent user load."""
        target_users = requirements['peak_concurrent_users']
        
        for instance_name, specs in self.aws_services['ec2_compute'].items():
            if specs['concurrent_users'] >= target_users * 1.1:  # 10% buffer
                return instance_name, specs['cost_monthly']
        
        # If no single instance fits, suggest the largest (would need load balancing)
        largest = max(self.aws_services['ec2_compute'].items(),
                     key=lambda x: x[1]['concurrent_users'])
        return largest[0], largest[1]['cost_monthly']
    
    def _optimize_redis_selection(self, requirements: Dict) -> Tuple[str, float]:
        """Select optimal Redis instance for caching needs."""
        cache_memory_needed = requirements['cache_memory_gb']
        
        for instance_name, specs in self.aws_services['elasticache_redis'].items():
            if specs['memory_gb'] >= cache_memory_needed * 1.5:  # 50% buffer for Redis overhead
                return instance_name, specs['cost_monthly']
        
        # Default to smallest if very little cache needed
        if cache_memory_needed < 1:
            return 'cache.t3.micro', self.aws_services['elasticache_redis']['cache.t3.micro']['cost_monthly']
        
        # Return largest if needed
        largest = max(self.aws_services['elasticache_redis'].items(),
                     key=lambda x: x[1]['memory_gb'])
        return largest[0], largest[1]['cost_monthly']
    
    def _calculate_additional_service_costs(self, analysis: Dict) -> Dict[str, float]:
        """Calculate costs for additional AWS services."""
        operations = analysis['operations']
        
        costs = {}
        
        # SES Email costs
        email_volume = operations['qr_email_phase']['emails_to_send']
        costs['ses'] = math.ceil(email_volume / 1000) * self.additional_services['ses_email']['cost_per_1000_emails']
        
        # WAF costs (estimated based on traffic)
        costs['waf'] = self.additional_services['waf']['typical_monthly']
        
        # S3 Storage costs
        storage_gb = operations['qr_email_phase']['storage_for_qrs_gb']
        costs['s3'] = storage_gb * self.additional_services['s3_storage']['cost_per_gb_month']
        
        # CloudFront CDN
        costs['cloudfront'] = self.additional_services['cloudfront_cdn']['typical_monthly']
        
        # Route53 DNS
        costs['route53'] = self.additional_services['route53_dns']['typical_monthly']
        
        # Lambda functions
        costs['lambda'] = self.additional_services['lambda_functions']['typical_monthly']
        
        # Netlify (external)
        # Choose plan based on traffic
        peak_users = analysis['requirements']['peak_concurrent_users']
        if peak_users > 1000:
            costs['netlify'] = self.additional_services['netlify_hosting']['business_plan']
        elif peak_users > 100:
            costs['netlify'] = self.additional_services['netlify_hosting']['pro_plan']
        else:
            costs['netlify'] = self.additional_services['netlify_hosting']['starter_plan']
        
        costs['total'] = sum(costs.values())
        
        return costs
    
    def _determine_salary_tier(self, requirements: Dict, aws_cost: float) -> Dict[str, float]:
        """Determine appropriate salary tier based on infrastructure complexity."""
        # Simple heuristic: base salary on AWS complexity and user load
        if aws_cost > 800 or requirements['peak_concurrent_users'] > 5000:
            return {'tier': 'Máximo', 'cost': 3000.00}
        elif aws_cost > 400 or requirements['peak_concurrent_users'] > 2000:
            return {'tier': 'Intenso', 'cost': 1800.00}
        elif aws_cost > 150 or requirements['peak_concurrent_users'] > 500:
            return {'tier': 'Moderado', 'cost': 1200.00}
        else:
            return {'tier': 'Básico', 'cost': 900.00}
    
    def _generate_optimization_notes(self, analysis: Dict) -> List[str]:
        """Generate specific optimization recommendations."""
        notes = []
        operations = analysis['operations']
        requirements = analysis['requirements']
        
        # Email optimization
        email_volume = operations['qr_email_phase']['emails_to_send']
        if email_volume > 10000:
            notes.append(f":material/email: Consider SES bulk sending for {email_volume:,} emails to reduce costs")
        
        # QR generation optimization
        qr_count = operations['qr_email_phase']['qr_generations_needed']
        if qr_count > 5000:
            notes.append(f":material/link: Use Lambda for QR generation ({qr_count:,} QRs) - batch process during low-traffic hours")
        
        # Scanning optimization
        peak_scans = operations['event_day_scanning']['peak_scans_per_hour']
        if peak_scans > 500:
            notes.append(f":material/flash_on: Redis caching critical for {peak_scans:,} scans/hour - consider read replicas")
        
        # Database optimization
        if requirements['peak_db_connections'] > 100:
            notes.append(":material/storage: Consider RDS Proxy to manage connection pooling efficiently")
        
        # Latency optimization
        if requirements['latency_target_ms'] < 200:
            notes.append(":material/rocket_launch: Multi-AZ deployment and CloudFront recommended for <200ms latency")
        
        # Cost optimization
        if requirements['peak_concurrent_users'] < 100:
            notes.append(":material/attach_money: Consider Aurora Serverless for variable workloads")
        
        return notes
    
    def recommend_infrastructure(self, events: List[Dict], latency_priority: bool = False) -> Dict[str, Any]:
        """Main recommendation function with enhanced analysis."""
        
        # Perform deep operational analysis
        analysis = self.analyze_operational_requirements(events)
        
        if not analysis:
            return self._get_default_recommendation()
        
        requirements = analysis['requirements']
        
        # Check if any predefined tier fits well
        best_predefined_tier = self._find_best_predefined_tier(requirements)
        
        # Generate custom infrastructure recommendation
        custom_config = self.generate_custom_infrastructure(analysis)
        
        # Determine if custom is better than predefined
        use_custom = self._should_use_custom_config(best_predefined_tier, custom_config, latency_priority)
        
        if use_custom:
            recommendation = {
                'type': 'custom',
                'config': custom_config,
                'analysis': analysis,
                'confidence': 'High - Custom optimized configuration',
                'cost_comparison': self._compare_with_predefined_tiers(custom_config, requirements),
                'latency_optimization': self._get_latency_recommendations(analysis, latency_priority)
            }
        else:
            recommendation = {
                'type': 'predefined',
                'tier_name': best_predefined_tier['name'],
                'config': best_predefined_tier,
                'analysis': analysis,
                'confidence': f"Medium - {best_predefined_tier['name']} tier fits requirements",
                'cost_comparison': {},
                'latency_optimization': self._get_latency_recommendations(analysis, latency_priority)
            }
        
        return recommendation
    
    def _find_best_predefined_tier(self, requirements: Dict) -> Dict[str, Any]:
        """Find the best fitting predefined tier."""
        for tier_name, tier_specs in self.predefined_tiers.items():
            if (requirements['total_events'] <= tier_specs['max_events'] and
                requirements['peak_concurrent_users'] <= tier_specs['max_concurrent_users']):
                
                # Get corresponding AWS costs (simplified mapping)
                aws_cost_mapping = {'Básico': 64, 'Moderado': 245, 'Intenso': 735, 'Máximo': 3166}
                
                return {
                    'name': tier_name,
                    'specs': tier_specs,
                    'aws_cost': aws_cost_mapping[tier_name],
                    'total_cost': aws_cost_mapping[tier_name] + tier_specs['salary_cost']
                }
        
        # If no tier fits, return the largest
        return {
            'name': 'Máximo',
            'specs': self.predefined_tiers['Máximo'],
            'aws_cost': 3166,
            'total_cost': 3166 + 3000
        }
    
    def _should_use_custom_config(self, predefined: Dict, custom: Dict, latency_priority: bool) -> bool:
        """Determine whether to use custom configuration over predefined tier."""
        # Use custom if it's significantly cheaper (>20% savings)
        cost_savings = predefined['total_cost'] - custom['total_monthly_cost']
        significant_savings = cost_savings > predefined['total_cost'] * 0.20
        
        # Use custom if predefined tier is overkill (>50% unused capacity)
        # This is a simplified check - in practice would be more nuanced
        
        # Use custom if latency is priority and custom provides better latency setup
        if latency_priority:
            return True
        
        # Use custom if significant cost savings
        return significant_savings
    
    def _compare_with_predefined_tiers(self, custom_config: Dict, requirements: Dict) -> Dict[str, Any]:
        """Compare custom configuration with all predefined tiers."""
        comparison = {}
        aws_cost_mapping = {'Básico': 64, 'Moderado': 245, 'Intenso': 735, 'Máximo': 3166}
        
        for tier_name, tier_specs in self.predefined_tiers.items():
            fits = (requirements['total_events'] <= tier_specs['max_events'] and
                   requirements['peak_concurrent_users'] <= tier_specs['max_concurrent_users'])
            
            total_cost = aws_cost_mapping[tier_name] + tier_specs['salary_cost']
            savings = total_cost - custom_config['total_monthly_cost']
            
            comparison[tier_name] = {
                'fits_requirements': fits,
                'monthly_cost': total_cost,
                'savings_vs_custom': savings,
                'percentage_savings': (savings / total_cost * 100) if total_cost > 0 else 0
            }
        
        return comparison
    
    def _get_latency_recommendations(self, analysis: Dict, latency_priority: bool) -> List[str]:
        """Get specific latency optimization recommendations."""
        recommendations = []
        requirements = analysis['requirements']
        
        if latency_priority or requirements['latency_target_ms'] < 500:
            recommendations.append(":material/public: Use multiple AWS regions for global latency optimization")
            recommendations.append(":material/flash_on: Enable CloudFront CDN with aggressive caching policies")
            recommendations.append(":material/refresh: Implement Redis cluster for sub-100ms cache responses")
            
        if requirements['peak_concurrent_users'] > 1000:
            recommendations.append(":material/balance: Use Application Load Balancer with least-latency routing")
            recommendations.append(":material/bar_chart: Enable detailed CloudWatch monitoring for latency tracking")
        
        if analysis['operations']['event_day_scanning']['peak_scans_per_hour'] > 1000:
            recommendations.append(":material/confirmation_number: Pre-cache ticket validation data before event start")
            recommendations.append(":material/smartphone: Use DynamoDB for ultra-fast ticket lookups during scanning")
        
        return recommendations
    
    def _get_default_recommendation(self) -> Dict[str, Any]:
        """Return default recommendation when no events provided."""
        return {
            'type': 'predefined',
            'tier_name': 'Básico',
            'config': {
                'name': 'Básico',
                'specs': self.predefined_tiers['Básico'],
                'aws_cost': 64,
                'total_cost': 964
            },
            'analysis': {},
            'confidence': 'Low - No events provided for analysis'
        }

    def generate_redis_cost_comparison(self, cache_memory_needed: float) -> pd.DataFrame:
        """Generate comparison between AWS ElastiCache and self-hosted Redis options."""
        comparison_data = []
        
        # AWS ElastiCache options
        for instance_name, specs in self.aws_services['elasticache_redis'].items():
            if specs['memory_gb'] >= cache_memory_needed * 0.5:  # Show relevant options
                comparison_data.append({
                    'Option': f"AWS {instance_name}",
                    'Type': 'Cloud (AWS)',
                    'Memory (GB)': f"{specs['memory_gb']:.1f}",
                    'Monthly Cost': f"${specs['cost_monthly']:,.2f}",
                    'Setup Time': 'Minutes',
                    'Maintenance': 'AWS Managed',
                    'Redundancy': 'Multi-AZ Available',
                    'Risk Level': 'Very Low'
                })
        
        # Self-hosted options
        for option_key, option_specs in self.self_hosted_redis.items():
            if option_specs['memory_gb'] >= cache_memory_needed * 0.8:  # Show relevant options
                comparison_data.append({
                    'Option': option_specs['name'],
                    'Type': 'Self-Hosted',
                    'Memory (GB)': f"{option_specs['memory_gb']}",
                    'Monthly Cost': f"${option_specs['monthly_total_cost']:,.2f}",
                    'Setup Time': '1-2 days',
                    'Maintenance': 'Self-managed',
                    'Redundancy': 'Single point of failure',
                    'Risk Level': 'Medium (cache only)'
                })
        
        return pd.DataFrame(comparison_data)
    
    def calculate_redis_savings_analysis(self, cache_memory_needed: float, campaign_duration_months: int = 12) -> Dict:
        """Calculate detailed savings analysis for self-hosted vs AWS Redis."""
        
        # Find best AWS option
        aws_option = None
        for instance_name, specs in self.aws_services['elasticache_redis'].items():
            if specs['memory_gb'] >= cache_memory_needed * 1.5:
                aws_option = specs
                break
        
        if not aws_option:
            aws_option = list(self.aws_services['elasticache_redis'].values())[-1]
        
        # Find best self-hosted option
        self_hosted_option = None
        for option_specs in self.self_hosted_redis.values():
            if option_specs['memory_gb'] >= cache_memory_needed * 0.8:
                self_hosted_option = option_specs
                break
        
        if not self_hosted_option:
            self_hosted_option = list(self.self_hosted_redis.values())[0]
        
        # Calculate costs over campaign duration
        aws_total_cost = aws_option['cost_monthly'] * campaign_duration_months
        self_hosted_total_cost = self_hosted_option['monthly_total_cost'] * campaign_duration_months
        
        savings = aws_total_cost - self_hosted_total_cost
        savings_percentage = (savings / aws_total_cost * 100) if aws_total_cost > 0 else 0
        
        return {
            'aws_option': aws_option,
            'self_hosted_option': self_hosted_option,
            'campaign_duration_months': campaign_duration_months,
            'aws_total_cost': aws_total_cost,
            'self_hosted_total_cost': self_hosted_total_cost,
            'total_savings': savings,
            'savings_percentage': savings_percentage,
            'break_even_months': self_hosted_option['hardware_cost_usd'] / (aws_option['cost_monthly'] - self_hosted_option['monthly_electricity_cost']) if aws_option['cost_monthly'] > self_hosted_option['monthly_electricity_cost'] else 0
        }

    def generate_detailed_cost_breakdown(self, recommendation: Dict) -> pd.DataFrame:
        """Generate detailed cost breakdown for the recommendation."""
        if recommendation['type'] == 'custom':
            config = recommendation['config']
            components = config['aws_components']
            
            breakdown_data = []
            
            # AWS Components
            breakdown_data.append({
                'Service': 'RDS Database',
                'Instance/Plan': components['rds']['instance'],
                'Monthly Cost': f"${components['rds']['cost']:,.2f}",
                'Category': 'AWS Core',
                'Description': 'PostgreSQL database for ticket and user data'
            })
            
            breakdown_data.append({
                'Service': 'EC2 Compute',
                'Instance/Plan': components['ec2']['instance'],
                'Monthly Cost': f"${components['ec2']['cost']:,.2f}",
                'Category': 'AWS Core',
                'Description': 'Main application server'
            })
            
            breakdown_data.append({
                'Service': 'ElastiCache Redis',
                'Instance/Plan': components['redis']['instance'],
                'Monthly Cost': f"${components['redis']['cost']:,.2f}",
                'Category': 'AWS Core',
                'Description': 'Caching for fast ticket validation'
            })
            
            # Additional Services
            additional = components['additional_services']
            for service, cost in additional.items():
                if service != 'total' and cost > 0:
                    service_names = {
                        'ses': 'SES Email Service',
                        'waf': 'Web Application Firewall',
                        's3': 'S3 Object Storage',
                        'cloudfront': 'CloudFront CDN',
                        'route53': 'Route53 DNS',
                        'lambda': 'Lambda Functions',
                        'netlify': 'Netlify Frontend'
                    }
                    
                    category = 'External' if service == 'netlify' else 'AWS Additional'
                    
                    breakdown_data.append({
                        'Service': service_names.get(service, service),
                        'Instance/Plan': 'Standard',
                        'Monthly Cost': f"${cost:,.2f}",
                        'Category': category,
                        'Description': f"{service.upper()} service costs"
                    })
            
            # Salary costs
            breakdown_data.append({
                'Service': 'Staff Salary',
                'Instance/Plan': 'Engineering Team',
                'Monthly Cost': f"${config['salary_cost']:,.2f}",
                'Category': 'Human Resources',
                'Description': 'Technical staff and support'
            })
            
            return pd.DataFrame(breakdown_data)
        
        else:
            # Predefined tier breakdown
            config = recommendation['config']
            breakdown_data = [
                {
                    'Service': 'AWS Infrastructure',
                    'Instance/Plan': f"{config['name']} Tier",
                    'Monthly Cost': f"${config['aws_cost']:,.2f}",
                    'Category': 'AWS Core',
                    'Description': f"Pre-configured {config['name']} tier infrastructure"
                },
                {
                    'Service': 'Staff Salary',
                    'Instance/Plan': 'Engineering Team',
                    'Monthly Cost': f"${config['specs'].get('salary_cost', 0):,.2f}",
                    'Category': 'Human Resources',
                    'Description': 'Technical staff and support'
                }
            ]
            
            return pd.DataFrame(breakdown_data) 