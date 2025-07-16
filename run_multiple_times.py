#!/usr/bin/env python3
"""
Script to run a command multiple times with configurable parameters.
Usage: python run_multiple_times.py --command "your_command" --times 5
"""

import argparse
import subprocess
import time
import sys
from datetime import datetime

def run_command(command, times=1, delay=0, verbose=True, save_output=False):
    """
    Run a command multiple times.
    
    Args:
        command (str): The command to run
        times (int): Number of times to run the command
        delay (float): Delay between runs in seconds
        verbose (bool): Whether to print detailed output
        save_output (bool): Whether to save output to files
    """
    results = []
    
    for i in range(times):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Run {i+1}/{times} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Command: {command}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run the command without capturing output (prints directly to terminal)
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=False,  # Don't capture output in memory
                text=True,
                timeout=None  # 无超时限制
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results (without stdout/stderr since we're not capturing)
            run_result = {
                'run': i + 1,
                'return_code': result.returncode,
                'stdout': '',  # Empty since we're not capturing
                'stderr': '',  # Empty since we're not capturing
                'duration': duration,
                'success': result.returncode == 0
            }
            results.append(run_result)
            
            # Print only start and end results
            if verbose:
                print(f"Return code: {result.returncode}")
                print(f"Duration: {duration:.2f} seconds")
                
                if result.returncode == 0:
                    print(" SUCCESS")
                else:
                    print(" FAILED")
                    # Note: stderr is not captured, so we can't show it
            
            # Save output to file if requested (but we don't have output to save)
            if save_output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"run_{i+1:03d}_{timestamp}.txt"
                
                with open(filename, 'w') as f:
                    f.write(f"Command: {command}\n")
                    f.write(f"Run: {i+1}/{times}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Duration: {duration:.2f} seconds\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write("\n" + "="*50 + "\n")
                    f.write("Note: Output was not captured (printed directly to terminal)\n")
                
                if verbose:
                    print(f"Run info saved to: {filename}")
            
        except subprocess.TimeoutExpired:
            if verbose:
                print(" TIMEOUT (should not happen with timeout=None)")
            results.append({
                'run': i + 1,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Unexpected timeout occurred',
                'duration': time.time() - start_time,
                'success': False
            })
        
        except Exception as e:
            if verbose:
                print(f" ERROR: {str(e)}")
            results.append({
                'run': i + 1,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': time.time() - start_time,
                'success': False
            })
        
        # Delay between runs
        if delay > 0 and i < times - 1:
            if verbose:
                print(f"Waiting {delay} seconds before next run...")
            time.sleep(delay)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_runs = sum(1 for r in results if r['success'])
    total_duration = sum(r['duration'] for r in results)
    avg_duration = total_duration / len(results) if results else 0
    
    print(f"Total runs: {times}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {times - successful_runs}")
    print(f"Success rate: {successful_runs/times*100:.1f}%")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Average duration: {avg_duration:.2f} seconds")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run a command multiple times')
    parser.add_argument('--command', '-c', required=True, help='Command to run')
    parser.add_argument('--times', '-t', type=int, default=1, help='Number of times to run (default: 1)')
    parser.add_argument('--delay', '-d', type=float, default=0, help='Delay between runs in seconds (default: 0)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    parser.add_argument('--save-output', '-s', action='store_true', help='Save output to files')
    
    args = parser.parse_args()
    
    if args.times < 1:
        print("Error: Number of times must be at least 1")
        sys.exit(1)
    
    print(f"Running command '{args.command}' {args.times} times")
    if args.delay > 0:
        print(f"Delay between runs: {args.delay} seconds")
    
    results = run_command(
        command=args.command,
        times=args.times,
        delay=args.delay,
        verbose=not args.quiet,
        save_output=args.save_output
    )
    
    # Exit with error if any run failed
    if not all(r['success'] for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main() 