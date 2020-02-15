
def get_from_metric(latest_metrics, query_metric_name, current_stage):
    for metric_stage_name, metric_value in latest_metrics.keys():
        target_metric_name = metric_stage_name.replace('{}_'.format(current_stage), "")
        if target_metric_name == current_stage:
            return metric_value
    else:
        return None
