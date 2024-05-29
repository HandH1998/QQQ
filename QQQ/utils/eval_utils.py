import logging
import fnmatch


logger = logging.getLogger("QQQ")

# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def update_results(results, new_result):
    for key, value in new_result.items():
        if key in results:
            results[key].update(value)
        else:
            results.update({key: value})

