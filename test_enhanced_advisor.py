#!/usr/bin/env python3

"""
Simple test script for the enhanced infrastructure advisor
"""

import sys
import os
from datetime import datetime, date

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from enhanced_infrastructure_advisor import EnhancedAWSInfrastructureAdvisor
    
    print("✅ Enhanced advisor imported successfully")
    
    # Create advisor instance
    advisor = EnhancedAWSInfrastructureAdvisor()
    print("✅ Advisor instance created")
    
    # Test with sample events
    events = [
        {
            'name': 'Small Concert',
            'num_tickets': 500,
            'occupancy_rate': 80,
            'event_date': date.today()
        },
        {
            'name': 'Medium Concert',
            'num_tickets': 2000,
            'occupancy_rate': 75,
            'event_date': date.today()
        }
    ]
    
    print(f"✅ Created {len(events)} test events")
    
    # Test operational analysis
    analysis = advisor.analyze_operational_requirements(events)
    print("✅ Operational analysis completed")
    print(f"   - Peak concurrent users: {analysis['requirements']['peak_concurrent_users']}")
    print(f"   - Email volume: {analysis['requirements']['email_volume']}")
    print(f"   - Total events: {analysis['requirements']['total_events']}")
    
    # Test infrastructure recommendation
    recommendation = advisor.recommend_infrastructure(events, latency_priority=False)
    print(f"✅ Infrastructure recommendation completed")
    print(f"   - Type: {recommendation['type']}")
    print(f"   - Confidence: {recommendation['confidence']}")
    
    if recommendation['type'] == 'custom':
        config = recommendation['config']
        print(f"   - Custom config monthly cost: ${config['total_monthly_cost']:.2f}")
        print(f"   - AWS cost: ${config['total_aws_cost']:.2f}")
        print(f"   - Components: {list(config['aws_components'].keys())}")
    else:
        config = recommendation['config']
        print(f"   - Recommended tier: {config['name']}")
        print(f"   - Monthly cost: ${config['total_cost']:.2f}")
    
    # Test with latency priority
    latency_recommendation = advisor.recommend_infrastructure(events, latency_priority=True)
    print(f"✅ Latency-priority recommendation completed")
    print(f"   - Type: {latency_recommendation['type']}")
    
    # Test cost breakdown
    if recommendation['type'] == 'custom':
        breakdown = advisor.generate_detailed_cost_breakdown(recommendation)
        print(f"✅ Cost breakdown generated: {len(breakdown)} services")
    
    print("\n🎉 All tests passed! Enhanced advisor is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 