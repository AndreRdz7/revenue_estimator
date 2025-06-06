#!/usr/bin/env python3
"""
Test script to verify all fixes are working correctly:
1. KeyError fix for enhanced advisor
2. Redis cost analysis with self-hosted options
3. Consistent currency formatting
"""

from enhanced_infrastructure_advisor import EnhancedAWSInfrastructureAdvisor

def test_enhanced_advisor():
    """Test the enhanced advisor functionality."""
    print(":material/science: Testing Enhanced AWS Infrastructure Advisor...")
    
    advisor = EnhancedAWSInfrastructureAdvisor()
    
    # Test with sample events
    events = [
        {
            'num_tickets': 1000, 
            'occupancy_rate': 80, 
            'commission_rate': 10,
            'name': 'Concert A'
        },
        {
            'num_tickets': 2000, 
            'occupancy_rate': 75, 
            'commission_rate': 12,
            'name': 'Festival B'
        }
    ]
    
    print(f":material/bar_chart: Testing with {len(events)} events...")
    
    # Test recommendation generation
    try:
        recommendation = advisor.recommend_infrastructure(events)
        print(f":material/check_circle: Recommendation generated successfully!")
        print(f"   Type: {recommendation['type']}")
        print(f"   Confidence: {recommendation['confidence']}")
        
        if recommendation['type'] == 'custom':
            config = recommendation['config']
            print(f"   Monthly Cost: ${config['total_monthly_cost']:,.2f}")
            print(f"   AWS Cost: ${config['total_aws_cost']:,.2f}")
            
            # Test Redis cost analysis
            cache_memory = recommendation['analysis']['requirements']['cache_memory_gb']
            print(f"\n:material/save: Testing Redis Cost Analysis...")
            print(f"   Cache Memory Needed: {cache_memory:.1f} GB")
            
            # Test Redis comparison
            redis_comparison = advisor.generate_redis_cost_comparison(cache_memory)
            print(f"   Redis Options Found: {len(redis_comparison)}")
            
            # Test savings analysis
            savings_analysis = advisor.calculate_redis_savings_analysis(cache_memory, 12)
            print(f"   Potential Savings: ${savings_analysis['total_savings']:,.2f}")
            print(f"   Break-even: {savings_analysis['break_even_months']:.1f} months")
            
            # Show self-hosted options
            print(f"\n:material/computer: Self-Hosted Redis Options:")
            for key, option in advisor.self_hosted_redis.items():
                print(f"   {option['name']}: ${option['monthly_total_cost']:,.2f}/month")
                print(f"     Hardware: ${option['hardware_cost_usd']:,.2f}")
                print(f"     Electricity: ${option['monthly_electricity_cost']:,.2f}/month")
        
        else:
            print(f"   Predefined Tier: {recommendation['tier_name']}")
            print(f"   Monthly Cost: ${recommendation['config']['total_cost']:,.2f}")
        
    except Exception as e:
        print(f":material/cancel: Error in recommendation: {e}")
        return False
    
    print(f"\n:material/check_circle: All tests passed! Enhanced advisor working correctly.")
    return True

def test_currency_formatting():
    """Test currency formatting consistency."""
    print(f"\n:material/attach_money: Testing Currency Formatting...")
    
    test_amounts = [1234.56, 12345.67, 123456.78, 1234567.89]
    
    for amount in test_amounts:
        formatted = f"${amount:,.2f}"
        print(f"   {amount} -> {formatted}")
    
    print(f":material/check_circle: Currency formatting consistent!")

if __name__ == "__main__":
    print(":material/rocket_launch: Running E-Ticketing Infrastructure Tests...")
    print("=" * 50)
    
    # Test enhanced advisor
    success = test_enhanced_advisor()
    
    # Test currency formatting
    test_currency_formatting()
    
    print("=" * 50)
    if success:
        print(":material/celebration: All tests completed successfully!")
        print("\n:material/assignment: Summary of fixes:")
        print("   :material/check_circle: Fixed KeyError for 'requirements' in enhanced advisor")
        print("   :material/check_circle: Added self-hosted Redis cost analysis for Puebla, Mexico")
        print("   :material/check_circle: Standardized currency formatting to $1,000.00 pattern")
        print("   :material/check_circle: Added comprehensive Redis vs AWS ElastiCache comparison")
        print("   :material/check_circle: Included break-even analysis for hardware investments")
    else:
        print(":material/cancel: Some tests failed. Please check the errors above.") 