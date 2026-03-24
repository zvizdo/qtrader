#!/usr/bin/env python3
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import re

def parse_iso8601(date_str):
    """
    Safely parse ISO 8601 date strings, handling the 'Z' suffix 
    which some Python versions don't natively parse with fromisoformat.
    """
    if not date_str:
        return None
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+00:00'
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        return None

def format_duration(seconds):
    if seconds == 0:
        return "0h"
    days = seconds / 86400
    if days >= 1:
        return f"{days:.2f} days"
    hours = seconds / 3600
    return f"{hours:.2f} hours"

def clean_currency(val_str):
    """
    Convert a currency string like '-$46.23' or '26.87%' to float.
    Handles percentages implicitly if treated as numbers, but we mostly use this for currency.
    """
    if isinstance(val_str, (int, float)):
        return float(val_str)
    
    cleaned = re.sub(r'[^\d.-]', '', str(val_str))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def analyze_directory(dir_path: Path, show_trades: bool = False):
    if not dir_path.is_dir():
        print(f"Error: Directory '{dir_path}' does not exist.")
        sys.exit(1)

    # Find LEAN JSON result: we exclude summary/order files to find the main output
    json_files = list(dir_path.glob("*.json"))
    valid_files = [
        f for f in json_files 
        if not f.name.endswith('-summary.json') 
        and not f.name.endswith('-order-events.json')
        and not f.name.startswith('data-monitor')
        and not f.name.startswith('machine-metric')
    ]
    
    if not valid_files:
        # Fallback to summary if main is missing
        summary_files = [f for f in json_files if f.name.endswith('-summary.json')]
        if summary_files:
            main_json_path = max(summary_files, key=lambda p: p.stat().st_size)
        else:
            print(f"Error: Could not find any valid backtest JSON file in {dir_path}")
            sys.exit(1)
    else:
        main_json_path = max(valid_files, key=lambda p: p.stat().st_size)
    
    try:
        with open(main_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {main_json_path}: {e}")
        sys.exit(1)
        
    stats = data.get("statistics", {})
    perf = data.get("totalPerformance", {})
    trade_stats = perf.get("tradeStatistics", {})
    closed_trades = perf.get("closedTrades", [])
    
    # High-Level Summary
    start_str = trade_stats.get("startDateTime")
    end_str = trade_stats.get("endDateTime")
    
    start_date = parse_iso8601(start_str)
    end_date = parse_iso8601(end_str)
    
    duration_days = (end_date - start_date).total_seconds() / 86400 if start_date and end_date else 0
    
    net_profit = stats.get("Net Profit", "0%")
    sharpe = stats.get("Sharpe Ratio", "0")
    drawdown = stats.get("Drawdown", "0%")
    
    # Trade & Hold Time Analysis
    total_trades = len(closed_trades)
    winning_trades = sum(1 for t in closed_trades if t.get("profitLoss", 0.0) > 0)
    win_rate = f"{(winning_trades / total_trades * 100):.0f}%" if total_trades > 0 else "0%"
    
    trade_durations_sec = []
    trade_pnls = []
    total_fees_from_trades = 0.0
    
    for t in closed_trades:
        entry = parse_iso8601(t.get("entryTime"))
        exit_ = parse_iso8601(t.get("exitTime"))
        if entry and exit_:
            duration = (exit_ - entry).total_seconds()
            trade_durations_sec.append(duration)
        trade_pnls.append(t.get("profitLoss", 0.0))
        total_fees_from_trades += t.get("totalFees", 0.0)
        
    avg_dur_sec = min_dur_sec = max_dur_sec = median_dur_sec = 0.0
    avg_pnl = 0.0
    
    total_long_sec = 0.0
    total_short_sec = 0.0
    for t, dur in zip(closed_trades, trade_durations_sec):
        if t.get("direction") == 0:
            total_long_sec += dur
        elif t.get("direction") == 1:
            total_short_sec += dur
            
    total_duration_sec = duration_days * 86400
    total_flat_sec = max(0.0, total_duration_sec - total_long_sec - total_short_sec)
    
    if trade_durations_sec:
        avg_dur_sec = sum(trade_durations_sec) / len(trade_durations_sec)
        min_dur_sec = min(trade_durations_sec)
        max_dur_sec = max(trade_durations_sec)
        sorted_durs = sorted(trade_durations_sec)
        mid = len(sorted_durs) // 2
        if len(sorted_durs) % 2 == 0:
            median_dur_sec = (sorted_durs[mid - 1] + sorted_durs[mid]) / 2.0
        else:
            median_dur_sec = sorted_durs[mid]
            
        avg_pnl = sum(trade_pnls) / len(trade_pnls)
    
    # Cost & Friction
    runtime = data.get("runtimeStatistics", {})
    abs_net_profit_str = runtime.get("Net Profit", "$0")
    abs_fees_str = runtime.get("Fees", "$0")
    
    abs_net_profit = clean_currency(abs_net_profit_str)
    
    # Fees are usually shown as negative in runtimeStats, or from trade list
    runtime_fees = abs(clean_currency(abs_fees_str))
    
    # Fallback to trade-level sum if runtime stats are missing/weird
    total_fees = runtime_fees if runtime_fees > 0 else total_fees_from_trades
    gross_profit = abs_net_profit + total_fees
    
    # Try capturing total fees from "statistics" dictionary as well
    if total_fees == 0 and "Total Fees" in stats:
        total_fees = abs(clean_currency(stats["Total Fees"]))
        gross_profit = abs_net_profit + total_fees

    # Formatted LLM-friendly Output
    lines = [
        "## Backtest Analysis Summary",
        "",
        "### 1. High-Level Summary",
        f"- **Start Date**: {start_date.strftime('%Y-%m-%d') if start_date else 'N/A'}",
        f"- **End Date**: {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}",
        f"- **Total Duration**: {duration_days:.2f} days",
        f"- **Total Net Profit**: {net_profit}",
        f"- **Sharpe Ratio**: {sharpe}",
        f"- **Max Drawdown**: {drawdown}",
        f"- **Win Rate**: {win_rate}",
        "",
        "### 2. Trade & Hold Time Analysis",
        f"- **Total Trades**: {total_trades}",
    ]
    
    if total_trades > 0:
        lines.extend([
            f"- **Total Time LONG**: {format_duration(total_long_sec)}",
            f"- **Total Time FLAT**: {format_duration(total_flat_sec)}",
            f"- **Avg Hold Duration**: {format_duration(avg_dur_sec)}",
            f"- **Median Hold**: {format_duration(median_dur_sec)}",
            f"- **Shortest Hold**: {format_duration(min_dur_sec)}",
            f"- **Longest Hold**: {format_duration(max_dur_sec)}",
            f"- **Avg PnL per Trade**: ${avg_pnl:.2f}"
        ])
    else:
        lines.append("- **Note**: No trades executed.")
        
    lines.extend([
        "",
        "### 3. Cost & Friction",
        f"- **Total Fees/Commissions**: ${total_fees:.2f}",
        f"- **Gross Profit**: ${gross_profit:.2f}",
        f"- **Net Profit**: ${abs_net_profit:.2f}"
    ])
    
    if show_trades and total_trades > 0:
        lines.extend([
            "",
            "### 4. Trade-by-Trade Analysis",
            "| Trade # | Entry Date | Time Since Last | Duration | PnL |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ])
        prev_exit = None
        for i, t in enumerate(closed_trades, 1):
            entry = parse_iso8601(t.get("entryTime"))
            exit_ = parse_iso8601(t.get("exitTime"))
            
            time_since_last = ""
            if prev_exit and entry:
                since_sec = (entry - prev_exit).total_seconds()
                # Use format_duration or just format in days as requested
                # format_duration handles hours, but to be strictly "days" if requested:
                # "time in days from previous trade"
                time_since_last = f"{since_sec / 86400:.2f} days"
                
            if entry and exit_:
                dur_sec = (exit_ - entry).total_seconds()
                dur_str = format_duration(dur_sec)
            else:
                dur_str = "N/A"
            pnl = t.get("profitLoss", 0.0)
            entry_str = entry.strftime('%Y-%m-%d %H:%M') if entry else "N/A"
            lines.append(f"| {i} | {entry_str} | {time_since_last} | {dur_str} | ${pnl:.2f} |")
            prev_exit = exit_

    print("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="Extract and format LEAN backtest results for LLM analysis.")
    parser.add_argument("--dir", required=True, help="Path to the LEAN backtest output directory")
    parser.add_argument("--show-trades", action="store_true", help="Include a trade-by-trade breakdown in the output")
    args = parser.parse_args()
    
    analyze_directory(Path(args.dir), show_trades=args.show_trades)

if __name__ == '__main__':
    main()
