# Tools

## monitor_log.py

This script could monitor the changes in log file.

- Use `tail` module to check timestamp of the log file and print the new added lines.
- Use `re` module to get the loss/acc infomathon from log file.
- Use `plot_log_curve` to plot the loss/acc curve(re-plot the canvas when log file updated).
