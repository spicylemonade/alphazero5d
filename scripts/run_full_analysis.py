"""
Master Script: Run Complete Performance and Learning Analysis
Executes all tests and generates comprehensive research report
"""
import subprocess
import sys
import time
from datetime import datetime
import json
import os

def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"{'=' * 80}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )

        elapsed = time.time() - start_time

        print(f"✓ Completed in {elapsed:.2f}s")
        print(result.stdout)

        return {
            'success': True,
            'duration': elapsed,
            'output': result.stdout,
            'error': None
        }

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        print(f"✗ Failed after {elapsed:.2f}s")
        print(f"Error: {e.stderr}")

        return {
            'success': False,
            'duration': elapsed,
            'output': e.stdout,
            'error': e.stderr
        }

def generate_summary_report(results, output_file):
    """Generate comprehensive summary report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': results,
        'summary': {
            'total_tests': len(results),
            'passed': sum(1 for r in results.values() if r['success']),
            'failed': sum(1 for r in results.values() if not r['success']),
            'total_duration': sum(r['duration'] for r in results.values())
        }
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report

def print_summary(report):
    """Print formatted summary"""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    summary = report['summary']
    print(f"\nTotal Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total Duration: {summary['total_duration']:.2f}s")

    print("\nTest Results:")
    for test_name, result in report['test_results'].items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {test_name}: {result['duration']:.2f}s")

    print("\n" + "=" * 80)

def main():
    print("=" * 80)
    print("5D CHESS AI - COMPLETE ANALYSIS SUITE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ensure directories exist
    os.makedirs('../results', exist_ok=True)

    # Define test suite
    tests = [
        {
            'name': 'Performance Testing',
            'command': 'cd ../tests && python test_performance.py',
            'description': 'Running gameplay performance tests'
        },
        {
            'name': 'Learning Analysis',
            'command': 'cd ../tests && python test_learning.py',
            'description': 'Analyzing learning effectiveness'
        },
        {
            'name': 'Architecture Optimization',
            'command': 'cd ../scripts && python optimize_architecture.py',
            'description': 'Optimizing architecture parameters'
        }
    ]

    results = {}

    # Run all tests
    for test in tests:
        result = run_command(test['command'], test['description'])
        results[test['name']] = result

        # Brief pause between tests
        time.sleep(2)

    # Generate summary report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"../results/analysis_summary_{timestamp}.json"

    report = generate_summary_report(results, summary_file)

    # Print summary
    print_summary(report)

    print(f"\nFull report saved to: {summary_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if report['summary']['failed'] == 0 else 1)

if __name__ == "__main__":
    main()
