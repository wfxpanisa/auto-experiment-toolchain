#!/usr/bin/python3

import sys
import os
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import multiprocessing as mp
import random

OP_FREE = 0
OP_MALLOC = 1
OP_CALLOC = 2
OP_REALLOC = 3


def generate_table_header(resize=False):
    result = [r'\begin{table}[h]', r'\centering']
    if resize:
        result.append(r'\resizebox{\textwidth}{!}{%')
    return result


def generate_table_footer(labelname, caption, resize=False):
    result = [r'\hline', r'\end{tabular}%']
    if resize:
        result.append(r'}')
    result.append(r'\caption{' + caption + '}')
    result.append(r'\label{tab:' + labelname + '}')
    result.append(r'\end{table}')
    return result


def create_size_summary(app, malloc_dfs):
    print('Running create_size_summary()..')

    total_allocation_amount = 0
    total_memory_allocated = 0
    avg_request_block_size = 0

    total_different_sizes = 0
    most_used_sizes = pd.Series(dtype='float64')
    share_most_used = 0

    for df_malloc in malloc_dfs:
        total_allocation_amount += df_malloc['REQ_SIZE'].count()
        total_memory_allocated += df_malloc['REQ_SIZE'].sum()
        avg_request_block_size += df_malloc['REQ_SIZE'].mean()

        total_different_sizes += df_malloc['REQ_SIZE'].value_counts(normalize=True).count()
        most_used_sizes = most_used_sizes.combine(df_malloc['REQ_SIZE'].value_counts()[:10], lambda x,y: x+y, fill_value=0)
        share_most_used += df_malloc['REQ_SIZE'].value_counts(normalize=True)[:10].sum()

    total_allocation_amount /= len(malloc_dfs)
    total_memory_allocated /= len(malloc_dfs)
    avg_request_block_size /= len(malloc_dfs)

    total_different_sizes /= len(malloc_dfs)
    most_used_sizes /= len(malloc_dfs)
    share_most_used /= len(malloc_dfs)

    most_used_sizes = most_used_sizes.sort_values(ascending=False).head(10)

    result1 = generate_table_header()
    result1.append(r'\begin{tabular}{| c | c |}')
    result1.append(r'\hline')
    result1.append(r'\multicolumn{2}{| c |}{\textbf{Allocation size summary}} \\')
    result1.append(r'\hline')
    result1.append(r'Total allocation count & {} \\'.format(int(total_allocation_amount)))
    result1.append(r'Total different sizes & {} \\'.format(int(total_different_sizes)))
    result1.append(r'Total allocated memory & {:.2f}Mb \\'.format(total_memory_allocated / 1024 / 1024))
    result1.append(r'Average block request size & {} bytes \\'.format(int(avg_request_block_size)))
    result1.append(r'Top 10 most used sizes total share & {:.2f}\% \\'.format(share_most_used * 100))
    result1 += generate_table_footer('totals' + app, f"Average counters for {app} results")

    result2 = generate_table_header(True)
    result2.append(r'\begin{tabular}{| c | c | c |}')
    result2.append(r'\hline')
    result2.append(
        r'\textbf{RAM Block Size Requested} & \textbf{Call count} & \textbf{\% of the total call amount} \\ ')
    result2.append(r'\hline')
    for k, v in most_used_sizes.items():
        result2.append(r"{} & {} & {:.2f}\% \\".format(k, int(v), v / total_allocation_amount * 100))
    result2 += generate_table_footer('sizes' + app, f"Request block size average counters for {app} results", True)
    return [result1, result2]


def create_graphs(app, df_malloc, df_free, base_export, error_list):
    print('Running create_graphs()..')

    mdf = get_concat_dataframes(df_malloc, df_free)

    fig, axes = plt.subplots(nrows=1, ncols=2)

    dfops = mdf['OP_ID'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    dfops = dfops.replace(0, "free").replace(1, "malloc").replace(2, "calloc").replace(3, "realloc")
    dfops = dfops.set_index('unique_values')
    dfops.plot.pie(
        y='counts',
        autopct='%1.1f%%',
        figsize=(7, 4),
        labels=None,
        ax=axes[0]).legend(labels=dfops.index, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12))

    axes[0].set_ylabel('')
    dt_aux = pd.DataFrame(error_list.items(), columns=['Error Name', 'Count'])
    dt_aux = dt_aux.set_index('Error Name')
    dt_aux = dt_aux[dt_aux['Count'] != 0]
    dt_aux.plot.pie(
        y='Count',
        autopct="%1.1f%%",
        figsize=(7, 4),
        labels=None,
        ax=axes[1]).legend(labels=dt_aux.index, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12))

    axes[1].set_ylabel('')

    filename = '{}_op_overview.pdf'.format(app)
    full_file_path = base_export + 'files/figures/' + filename
    fig.savefig(full_file_path, bbox_inches="tight")
    # filename = '{}_op_count.pdf'.format(app)
    # full_file_path = base_export + 'files/figures/' + filename
    # fig.savefig(full_file_path, bbox_inches="tight")
    result1 = [
        r'\begin{figure}[h]',
        r'\centering',
        r'\includegraphics{appendix/files/figures/' + filename + '}',
        r'\caption{Operation count on ' + app + '}{Source: Author}',
        r'\label{fig:opcount' + app + '}',
        '\end{figure}',
    ]

    #   result2 = []
    #   result2.append(r'\begin{figure}')
    #   result2.append(r'\centering')
    #   result2.append(r'\includegraphics{appendix/files/figures/' + filename + '}')
    #   result2.append(r'\caption{Valid operations share on '+ app +'}{Source: Author}')
    #   result2.append(r'\label{fig:validratio' + app + '}')
    #   result2.append('\end{figure}')
    return [result1]


def create_problem_summary(malloc_dfs, free_dfs, total_count):
    print('Running create_problem_summary()..')
    df_malloc = pd.concat([f for f in malloc_dfs], ignore_index=True)
    df_free = pd.concat([f for f in free_dfs], ignore_index=True)

    empty_alloc = df_malloc[(df_malloc['REQ_SIZE'] == 0) & ((df_malloc['OP_ID'] == 1) | (df_malloc['OP_ID'] == 2))].size
    huge_blocks = df_malloc[df_malloc['REQ_SIZE'] > 1E6].size
    return [r'{} & {}({:.2f}\%)'.format("Empty allocations", empty_alloc // len(malloc_dfs), 100*(empty_alloc // len(malloc_dfs))/total_count),
            r'{} & {}({:.2f}\%)'.format("Allocations > 1Mb", huge_blocks // len(malloc_dfs), 100*(huge_blocks // len(malloc_dfs))/total_count)]


def get_concat_dataframes(df_malloc, df_free):
    print('Running get_concat_dataframes()..')
    mdf = pd.concat([df_malloc, df_free], ignore_index=True)
    mdf.drop_duplicates(inplace=True)
    mdf.sort_values('TSTAMP', inplace=True)
    mdf.reset_index(drop=True, inplace=True)
    mdf["OP_ID"] = pd.to_numeric(mdf["OP_ID"])
    return mdf


def final_summary(mdf, app):
    #error_types = ['double free', 'invalid free', 'invalid malloc', 'invalid calloc', 'realloc freed',
    #               'invalid realloc', 'null free']

    error_types = ['null free', 'double free', 'invalid free', 'realloc freed', 'invalid realloc']
    error_list = {e: 0 for e in error_types}

    time_types = ['Short', 'Medium', 'Long']
    time_list = {e: 0 for e in time_types}

    ht = {}
    tt = {}
    ix = []
    ma = []
    allocated_memory = 0

    #memaddr = '0x171dd60'
    #print(mdf[(mdf.MEM_POS == memaddr) | (mdf.MEM_PTR == memaddr)]); sys.exit(0)

    for index, row in mdf.iterrows():

        op = row['OP_ID']
        ts = row['TSTAMP']
        mempos = row['MEM_POS']
        memptr = row['MEM_PTR']
        size = row['REQ_SIZE']

        if op == OP_FREE:
            if memptr == '(nil)':
                # This one is harmless
                error_list['null free'] += 1
                continue

            if memptr in ht:
                if ht[memptr] == OP_FREE:
                    # UB
                    error_list['double free'] += 1
                    continue
            else:
                # UB
                error_list['invalid free'] += 1
                continue

            delta_time = (ts - tt[memptr]) / 1E6
            if delta_time < 100:
                time_list['Short'] += 1
            elif delta_time < 1000:
                time_list['Medium'] += 1
            else:
                time_list['Long'] += 1

            ht[memptr] = OP_FREE

        elif op == OP_MALLOC:
            #if mempos in ht and ht[mempos] != OP_FREE:
            #    error_list['invalid malloc'] += 1
            #    continue
            ht[mempos] = OP_MALLOC
            tt[mempos] = ts
            allocated_memory += size

        elif op == OP_CALLOC:
            #if mempos in ht and ht[mempos] != OP_FREE:
            #    error_list['invalid calloc'] += 1
            #    continue
            ht[mempos] = OP_CALLOC
            tt[mempos] = ts
            allocated_memory += size

        elif op == OP_REALLOC:

            # act as malloc
            if memptr == '(nil)':
                ht[mempos] = OP_MALLOC
                #if mempos in ht and ht[mempos] != OP_FREE:
                #   error_list['invalid realloc'] += 1
                #   continue
            else:
                if memptr in ht:
                    if ht[memptr] == OP_FREE:
                        error_list['realloc freed'] += 1
                        continue
                else:
                    error_list['invalid realloc'] += 1
                    continue

                delta_time = (ts - tt[memptr]) / 1E6
                if delta_time < 100:
                    time_list['Short'] += 1
                elif delta_time < 1000:
                    time_list['Medium'] += 1
                else:
                    time_list['Long'] += 1

                ht[memptr] = OP_FREE

            if size > 0:
                ht[mempos] = OP_REALLOC
                tt[mempos] = ts
                allocated_memory += size

        ix.append(index)
        ma.append(allocated_memory)

    df = pd.DataFrame({'index': ix, 'memory': ma})
    df['index'] = (df['index'] - df['index'][0]) / 1E6
    df['memory'] = df['memory'] / 1024 / 1024

    ax = df.plot(x='index', y='memory', grid=True, label="Allocation amount in millions", figsize=(8, 2), kind='line')
    ax.set_ylabel('Total Memory Usage (in Mb)')

    filename = '{}_memory_usage.pdf'.format(app)
    full_file_path = '/home/victor/Coding/msc_exp/master-thesis-draft/appendix/files/figures/' + filename
    ax.figure.savefig(full_file_path)

    result = [r'\begin{figure}[h]',
              r'\centering',
              r'\includegraphics[width=\textwidth]{appendix/files/figures/' + filename + '}',
              r'\caption{Total memory usage on ' + app + ' (in Mb)}{Source: Author}',
              r'\label{fig:memoryusage' + app + '}',
              '\end{figure}']

    return ([r"{} & {}".format(k, v) for k, v in error_list.items()],
            [r"{} allocations & {}({:.2f}\%)".format(k, v, v*100/(sum(time_list.values()))) for k, v in time_list.items()],
            result,
            sum(time_list.values()),
            error_list)


def create_remarks_summary(malloc_dfs, free_dfs, app):
    print('Running create_remarks_summary()..')

    b_lines, t_lines, h_lines, alloc_count, error_list = final_summary(get_concat_dataframes(malloc_dfs[0], free_dfs[0]), app)
    pbm_count = sum(error_list.values())
    valid_count = len(malloc_dfs[0]) + len(free_dfs[0]) - pbm_count
    p_lines = create_problem_summary(malloc_dfs, free_dfs, alloc_count)

    result1 = generate_table_header()
    result1.append(r'\begin{tabular}{| c | c |}')
    result1.append(r'\hline')
    result1.append(r'\multicolumn{2}{| c |}{\textbf{Problematic Behavior Counts}} \\')
    result1.append(r'\hline')
    result1.append(b_lines[0] + r' \\')
    result1.append(b_lines[1] + r' \\')
    result1.append(b_lines[2] + r' \\')
    result1.append(b_lines[3] + r' \\')
    result1 += generate_table_footer('probehavior' + app, f"Potentially flawed behaviors on {app}")

    result2 = generate_table_header()
    result2.append(r'\begin{tabular}{| c | c |}')
    result2.append(r'\hline')
    result2.append(r'\multicolumn{2}{| c |}{\textbf{General Remarks}} \\')
    result2.append(r'\hline')
    result2.append(r"{} & {} \\".format("Total operation count", len(malloc_dfs[0]) + len(free_dfs[0])))
    result2.append(r"{} & {:.2f}\% \\".format("Valid operations", (1-(pbm_count/(len(malloc_dfs[0]) + len(free_dfs[0]))))*100))
    result2.append(t_lines[0] + r' \\')
    result2.append(t_lines[1] + r' \\')
    result2.append(t_lines[2] + r' \\')
    result2.append(p_lines[0] + r' \\')
    result2.append(p_lines[1] + r' \\')
    #result2.append(p_lines[2] + r' \\')
    result2 += generate_table_footer('genremarks' + app, f"General Remarks on {app} results")
    return [result1, result2, h_lines], error_list


def export_to_file(out, txt):
    with open(out, 'w') as f:
        f.write('\n'.join(txt))


def malloc_read(filename):
    malloc_file_columns = ['REQ_SIZE', 'TSTAMP', 'OP_ID', 'MEM_POS', 'MEM_PTR', 'ID_THREAD']
    return pd.read_csv(filename, names=malloc_file_columns, sep=';')


def free_read(filename):
    free_file_columns = ['TSTAMP', 'OP_ID', 'MEM_PTR', 'ID_THREAD']
    return pd.read_csv(filename, names=free_file_columns, sep=';')


def main(args):
    base_export = '/home/victor/Coding/msc_exp/master-thesis-draft/appendix/'

    csv_path = '/opt/exp_out/' + args[-1] + '/'
    app = csv_path.split('/')[-2]
    print(f'Starting {app} summary..')

    tmp_file_list = os.listdir(csv_path)

    with mp.Pool(processes=len(tmp_file_list))as pool:
        malloc_dfs = pool.map(malloc_read, [csv_path + f for f in tmp_file_list if f.startswith('malloc-')])
        free_dfs = pool.map(free_read, [csv_path + f for f in tmp_file_list if f.startswith('free-')])

    print('Setup complete..')

    list_floats = [create_size_summary(app, malloc_dfs)]
    list_item, error_list = create_remarks_summary(malloc_dfs, free_dfs, app)
    list_floats.append(list_item)
    list_floats.append(create_graphs(app, malloc_dfs[0], free_dfs[0], base_export, error_list))

    flat_list = [item for sublist in list_floats for item in sublist]
    txt = [r'\cleardoublepage']
    txt.append(r'\section{' + app + r'} \label{appendix:' + app + '}')
    for i in [0, 1, 2, 4, 3, 5]:
        txt.append('\n'.join(flat_list[i]))

    export_to_file(base_export + f'summaries/{app}.tex', txt)
    print(f"Finished {app} summary")


if __name__ == '__main__':
    main(sys.argv)
