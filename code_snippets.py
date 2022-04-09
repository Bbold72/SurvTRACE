runs_list.append(run)
file_name = 'survtrace' + '_' + STConfig['data'] + '_'+ str(time.time()) + '.pickle'
with open(Path('results/seer', file_name), 'wb') as f:
    pickle.dump(run, f)