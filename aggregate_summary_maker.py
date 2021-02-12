#!/usr/bin/python3
import sys
import os
import pathos.multiprocessing as mp
import dill
import matplotlib.pyplot as plt # type: ignore
import pandas as pd  # type: ignore
import numpy as np #type: ignore
import seaborn as sns #type: ignore
from scipy import stats # type: ignore
from statsmodels.stats.multicomp import pairwise_tukeyhsd # type: ignore
from matplotlib import cm

file_explorer = ['4pane', 'gentoo', 'pcmanfm', 'spacefm', 'tuxcmd', 'dolphin', 'konqueror', 'krusader', 'pcmanfm-qt', 'peony']

text_editor = ['bluefish', 'emacs-gtk', 'geany', 'gedit', 'mousepad', 'pluma', 'featherpad', 'juffed', 'kate', 'ktikz', 'kwrite', 'qtikz']

image_viewer = ['gimp', 'pinta', 'geeqie', 'gliv', 'gpicview', 'gthumb', 'viewnior', 'kolourpaint', 'photoflare', 'deepin-image-viewer', 'gwenview', 'lximage-qt', 'nomacs', 'phototonic']

music_player = ['rhythmbox', 'parole(music)', 'totem(music)', 'gmerlin(music)', 'elisa', 'kaffeine(music)', 'smplayer(music)', 'dragonplayer(music)']

video_player = ['parole(video)', 'totem(video)', 'gmerlin(video)', 'kaffeine(video)', 'smplayer(video)', 'dragonplayer(video)']

gtk = ['4pane', 'gentoo', 'pcmanfm', 'spacefm', 'tuxcmd', 'bluefish', 'emacs-gtk', 'geany', 'gedit', 'mousepad', 'pluma', 'gimp', 'pinta', 'geeqie', 'gliv', 'gpicview', 'gthumb', 'viewnior', 'rhythmbox', 'parole(music)', 'totem(music)', 'gmerlin(music)', 'parole(video)', 'totem(video)', 'gmerlin(video)']

qt = ['dolphin', 'konqueror', 'krusader', 'pcmanfm-qt', 'peony', 'featherpad', 'juffed', 'kate', 'ktikz', 'kwrite', 'qtikz', 'kolourpaint', 'photoflare', 'deepin-image-viewer', 'gwenview', 'lximage-qt', 'nomacs', 'phototonic', 'elisa', 'kaffeine(music)', 'smplayer(music)', 'dragonplayer(music)', 'kaffeine(video)', 'smplayer(video)', 'dragonplayer(video)']

all_apps = gtk + qt

try:
    memo_dfs = dill.load(open("df.dat", "rb"))
except:
    memo_dfs = {}

def get_app_toolkit(app):
    if app in gtk:
        return 'gtk'
    elif app in qt:
        return 'qt'
    else:
        return 'ERROR'

def get_app_category(app):
    if app in file_explorer:
        return 'file_explorer'
    elif app in text_editor:
        return 'text_editor'
    elif app in image_viewer:
        return 'image_viewer'
    elif app in music_player:
        return 'music_player'
    elif app in video_player:
        return 'video_player'
    else:
        return 'ERROR'


def memoize(f):
    def wrapper(*args, **kwargs):
        arg = args[0]
        name = f.__name__
        if name not in memo_dfs:
            memo_dfs[name] = {}
        if not arg in memo_dfs[name]:
            memo_dfs[name][arg] = f(*args, **kwargs)
            dill.dump(memo_dfs, open("df.dat", "wb"))
        return memo_dfs[name][arg]
    return wrapper

def malloc_read(filename):
    malloc_file_columns = ['REQ_SIZE', 'TSTAMP', 'OP_ID', 'MEM_POS', 'MEM_PTR', 'ID_THREAD']
    return pd.read_csv(filename, names=malloc_file_columns, sep=';')

def free_read(filename):
    malloc_file_columns = ['TSTAMP', 'OP_ID', 'MEM_PTR', 'ID_THREAD']
    return pd.read_csv(filename, names=malloc_file_columns, sep=';')

@memoize
def get_req_size_count_top10(app):
    print(app)
    csv_path = '/opt/exp_out/' + app + '/'
    tmp_file_list = os.listdir(csv_path)
    with mp.Pool(processes=len(tmp_file_list)) as pool:
        malloc_dfs = pool.map(malloc_read, [csv_path + f for f in tmp_file_list if f.startswith('malloc-')][:1])

    lst = []
    for df in malloc_dfs:
        for k, v in df['REQ_SIZE'].value_counts()[:10].items():
            lst.append([k, v, app, get_app_category(app), get_app_toolkit(app)])
    return pd.DataFrame(lst, columns = ['req_size', 'count', 'app', 'category', 'toolkit'])

@memoize
def get_req_size_count(app):
    csv_path = '/opt/exp_out/' + app + '/'
    tmp_file_list = os.listdir(csv_path)
    with mp.Pool(processes=len(tmp_file_list)) as pool:
        malloc_dfs = pool.map(malloc_read, [csv_path + f for f in tmp_file_list if f.startswith('malloc-')])

    lst = []
    for df in malloc_dfs:
        for k, v in df['REQ_SIZE'].value_counts().items():
            lst.append([k, v, app, get_app_category(app), get_app_toolkit(app)])
    return pd.DataFrame(lst, columns = ['req_size', 'count', 'app', 'category', 'toolkit'])

def anova_tukey(apps):

    df = get_share_df(apps)

    #df['combination'] = df.req_size.apply(str) + "/" + df.toolkit
    #print(df)

    data = {}
    for index, row in df.iterrows():
        if row['category'] not in data:
            data[row['category']] = []
        data[row['category']].append(float(row['share']))

    print(data)
    #a = df[df['toolkit'] == 'gtk']['share'].astype('float')
    #b = df[df['toolkit'] == 'qt']['share'].astype('float')
    #print(stats.ttest_ind(a,b))
    #sys.exit(0)
    # stats f_oneway functions takes the groups as input and returns F and P-value
    fvalue, pvalue = stats.f_oneway(*[data[k] for k in data])
    print(f"Results of ANOVA test:\n The F-statistic is: {fvalue}\n The p-value is: {pvalue}")

    # perform multiple pairwise comparison (Tukey HSD)
    #m_comp = pairwise_tukeyhsd(endog=df['share'], groups=df['category'], alpha=0.05)

    # coerce the tukeyhsd table to a DataFrame
    tukey_data = pd.DataFrame(data=m_comp._results_table.data[1:], columns = m_comp._results_table.data[0])
    group1_comp = tukey_data.loc[tukey_data.reject == True].groupby('group1').reject.count()
    group2_comp = tukey_data.loc[tukey_data.reject == True].groupby('group2').reject.count()
    tukey_data = pd.concat([group1_comp, group2_comp], axis=1)

    tukey_data = tukey_data.fillna(0)
    tukey_data.columns = ['reject1', 'reject2']
    tukey_data['total_sum'] = tukey_data.reject1 + tukey_data.reject2

    # just show the top 20 results
    print(tukey_data.sort_values('total_sum',ascending=False))
    m_comp.plot_simultaneous().savefig('/tmp/test.pdf')
    #print(m_comp.summary())

def retention_time_graph(apps):
    df = get_time_df(apps)
    df['count'] = df['count'].astype('float')
    fig, ax = plt.subplots()

    df = df.groupby(['retention', 'category']).mean().unstack('retention')
    df.columns = ['Long', 'Medium', 'Short']

    df = df.div(df.sum(1), axis=0) * 100

    df.iloc[::1].plot(kind='bar', ax=ax, stacked=False, rot=0)
    plt.minorticks_on()
    plt.yscale("log")
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("Application category")
    ax.set_ylabel("Allocation amount")

    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

def get_time_df(apps):
    l = []
    r = []
    q = []
    c = []
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            [l.append(line.split(' ')[-2].split('(')[0]) for line in f.readlines()[70:73]]
            r.append('short')
            r.append('medium')
            r.append('long')
            q.append(app)
            q.append(app)
            q.append(app)
            c.append(get_app_category(app))
            c.append(get_app_category(app))
            c.append(get_app_category(app))
    return pd.DataFrame({'count': l, 'category': q, 'retention': r, 'category': c})

def get_pbm_df(apps):
    l = []
    r = []
    q = []
    c = []
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            [l.append(line.split(' ')[-2].split('(')[0]) for line in f.readlines()[47:51]]
            r.append('null free')
            r.append('double free')
            r.append('invalid free')
            r.append('realloc freed')
            q.append(app)
            q.append(app)
            q.append(app)
            q.append(app)
            c.append(get_app_category(app))
            c.append(get_app_category(app))
            c.append(get_app_category(app))
            c.append(get_app_category(app))
    return pd.DataFrame({'count': l, 'app': q, 'pbm': r, 'category': c})

def top_sizes_share_graph(apps):

    df = get_share_df(apps)
    df.share = df.share.astype('float')
    print(np.std(df.share))
    df1 = df.groupby(['category']).mean()
    df2 = df.groupby(['toolkit']).mean()
    df = df1.append(df2)
    print(np.std(df.share))

    fig, ax= plt.subplots()
    df.sort_values(by='share').plot.bar(y='share', ax=ax, colormap='Accent', rot=0)

    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')


def get_share_df(apps):
    share = []
    appl = []
    cat = []
    tk = []
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            share.append(f.readlines()[12].split('&')[1].split('\\')[0])
            appl.append(app)
            cat.append(get_app_category(app))
            tk.append(get_app_toolkit(app))
    return pd.DataFrame({'share': share, 'app': appl, 'category': cat, 'toolkit': tk })

def operation_count_graph(apps):
    dfs = []
    for app in apps:
        dfs.append(get_op_df(app))
    df = pd.concat(dfs, ignore_index=True)
    #df.share = df.share.astype('float')
    #.value_counts(normalize=True)
    #df.plot.bar(y='share', ax=ax, colormap='Accent', rot=0)

    fig, ax= plt.subplots()
    df = df.groupby(['op_id', 'category']).mean().unstack('op_id')
    df.columns = ['free', 'malloc', 'calloc', 'realloc']
    df.plot(kind='bar', ax=ax, stacked=False, rot=0)

    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("Application category")
    ax.set_ylabel("TOC")

    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

def problem_graph(apps):
    df = get_pbm_df(apps)
    df['count'] = df['count'].astype('float')
    fig, ax= plt.subplots()
    df = df.groupby(['pbm', 'category']).mean().unstack('pbm')
    df.columns = ['double free', 'invalid free', 'null free', 'realloc freed']
    df.plot(kind='bar', ax=ax, stacked=False, rot=0)

    plt.minorticks_on()
    plt.yscale("log")
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("Application category")
    ax.set_ylabel("Operation count")

    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

@memoize
def get_op_df(app):
    csv_path = '/opt/exp_out/' + app + '/'
    tmp_file_list = os.listdir(csv_path)
    with mp.Pool(processes=len(tmp_file_list)) as pool:
        malloc_dfs = pool.map(malloc_read, [csv_path + f for f in tmp_file_list if f.startswith('malloc-')])
        free_dfs = pool.map(free_read, [csv_path + f for f in tmp_file_list if f.startswith('free-')])
    lst = []
    for df in malloc_dfs:
        for k, v in df['OP_ID'].value_counts().items():
            lst.append([k, v, app, get_app_category(app), get_app_toolkit(app)])
    for df in free_dfs:
        for k, v in df['OP_ID'].value_counts().items():
            lst.append([k, v, app, get_app_category(app), get_app_toolkit(app)])

    return pd.DataFrame(lst, columns =['op_id', 'count', 'app', 'category', 'toolkit'])


def diff_size_plot(args):
    dfs = []
    for app in args:
        dfs.append(get_req_size_count(app))
    df = pd.concat(dfs, ignore_index=True)
    ls = []
    rs = []
    for cat,_ in df.category.value_counts().items():
        ls.append(len(df[df['category'] == cat].req_size.value_counts()))
        rs.append(cat)
    df = pd.DataFrame({'count': ls, 'cat': rs})

    fig, ax= plt.subplots()
    df.sort_values(by='count').plot.bar(y='count', x='cat',ax=ax, colormap='Accent', rot=0)
    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("Application Category")
    ax.set_ylabel("TDRS")
    #ax.set_title(("Precipitation"), fontsize=20)
    ax.axhline(df["count"].median(), label='median')
    print(df["count"].median())
    ax.legend()
    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

def get_pbm_share_from_summary(apps):
    lst = []
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            lst.append([f.readlines()[69].split('&')[-1].split(' ')[1].split('\\')[0], app, get_app_category(app), get_app_toolkit(app)])
    return pd.DataFrame(lst, columns =['valid_share', 'app', 'category', 'toolkit'])

def pbm_share_group(apps):
    df = get_pbm_share_from_summary(apps)
    df['valid_share'] = df['valid_share'].astype('float')
    fig, ax= plt.subplots()
    df1 = df.groupby(['category']).mean()
    df2 = df.groupby(['toolkit']).mean()
    df3 = df1.append(df2)

    plot = df3.sort_values(by='valid_share').plot.bar(y='valid_share', yerr=df3['valid_share'].sem(), ax=ax, stacked=True, rot=45)
    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom='off')
    ax.set_xlabel("Application group")
    ax.set_ylabel("Valid operations %")
    ax.set_ylim(0, 70)
    ax.axhline(df3["valid_share"].median(), label='median', color='orange')
    ax.legend()
    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

def pbm_count(apps):
    #dr = get_total_ops(apps)
    df = get_pbm_df(apps)
    df['count'] = df['count'].astype('float')
    tl = []
    for key, value in df.items():
        tl.append((key, '{:.2f}'.format(df[df['app'] == key]['count'].sum() / float(value))))
    tl.sort(key=lambda x: x[1])
    print(sum([float(a[1]) for a in tl])/len(tl))

def get_total_ops(apps):
    a = []
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            r[app] = f.readlines()[68].split('&')[-1].split(' ')[1]
    df = pd.DataFrame({'count': ls, 'cat': rs})

def plot_corr(apps):
    a, b, c, d, e = [[] for i in range(5)]
    for app in apps:
        with open(f"/home/victor/Coding/msc_exp/master-thesis-draft/appendix/summaries/{app}.tex", "r") as f:
            f = f.readlines()
            a.append(int(f[68].split('&')[-1].split(' ')[1]))
            b.append(float(f[10].split('&')[-1].split('M')[0]))
            c.append(int(f[11].split('&')[-1].split(' ')[1]))
            d.append(int(f[9].split('&')[-1].split(' ')[1]))
            e.append(float(f[12].split('&')[-1].split('\\')[0]))

    df = pd.DataFrame(
            {
                'TOC': a,
                'TAM': b,
                'ARS': c,
                'TDRS': d,
                'TTMUSS': e
            }
        )
    corr_heatmap(df)

def corr_heatmap(df):
#figsize=(8, 8)
    f, ax = plt.subplots()
    dfc = df.corr(method='spearman')
    maskTriu = np.triu(dfc)
    for a in maskTriu:
        for i, _ in enumerate(a):
            if a[i] == 1:
                a[i] = 0
    s = sns.heatmap(dfc, mask=maskTriu, annot=True, cmap="YlGnBu", vmax=1, vmin=-1, center=0, square=False, linewidths=.5, cbar_kws={"shrink": .5, "orientation": "horizontal"})
    f.tight_layout()
    f.savefig('/tmp/test.pdf')

@memoize
def new_malloc_read(app):
    filename = filename = next((''.join(['/opt/exp_out/', app, '/', f]) for f in os.listdir(''.join(['/opt/exp_out/', app, '/'])) if f.startswith('malloc-')), None)
    df = malloc_read(filename)
    df = df.drop(columns=['TSTAMP', 'OP_ID', 'MEM_POS', 'MEM_PTR', 'ID_THREAD'])
    df['app'] = app
    df['category'] = get_app_category(app)
    df['toolkit'] = get_app_toolkit(app)
    return df

def boxplot(apps):

    #fig, axes = plt.subplots(nrows=3, ncols=2)
    fig, ax = plt.subplots()
    #for ix, apps in enumerate(blocks):
        #ax = axes[ix%3][ix%2]

    with mp.Pool(processes=len(apps)) as pool:
        dfs  = pool.map(new_malloc_read, apps)
    #dfs = []
    #for app in apps:
    #    dfs = append(new_malloc_read(app))

    #for df in dfs:
    #    print(df.describe(), df['app'][0])

    df = pd.concat(dfs, ignore_index=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.float_format', lambda x: '%.0f' % x):
        print(df.groupby('toolkit').describe().unstack(1))

    sns.boxplot(x='toolkit', y='REQ_SIZE', data=df, showfliers=False, ax=ax)
    #plt.minorticks_on()
    #ax.tick_params(axis='x', which='minor', bottom='off')
    plt.xticks(rotation=45)
    #plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel("")
    ax.set_ylabel("Request size")
    #ax.set_title(("Precipitation"), fontsize=20)
    #ax.axhline(df["count"].mean(), label='mean')
    #ax.legend()

    fig.tight_layout()
    fig.savefig('/tmp/test.pdf')

@memoize
def aux_overlap_func(app):

    OP_FREE = 0
    OP_MALLOC = 1
    OP_CALLOC = 2
    OP_REALLOC = 3

    basepath = f'/opt/exp_out/{app}'
    df_free, df_malloc = sorted(['/'.join([basepath, e]) for e in os.listdir(basepath) if '-1' in e])
    mdf = pd.concat([malloc_read(df_malloc), free_read(df_free)], ignore_index=True)
    mdf.drop_duplicates(inplace=True)
    mdf.sort_values('TSTAMP', inplace=True)
    mdf.reset_index(drop=True, inplace=True)
    mdf["OP_ID"] = pd.to_numeric(mdf["OP_ID"])

    ht = {}
    block_count = []
    block_amount = 0
    time_since_start = []

    mdf['TSTAMP'] -= mdf['TSTAMP'][0]

    for index, row in mdf.iterrows():
        op = row['OP_ID']
        ts = row['TSTAMP']
        mempos = row['MEM_POS']
        memptr = row['MEM_PTR']
        size = row['REQ_SIZE']

        if op == OP_FREE:
            if memptr == '(nil)':
                continue
            if memptr in ht:
                if ht[memptr] == OP_FREE:
                    continue
            else:
                continue
            ht[memptr] = OP_FREE
            block_amount -= 1

        elif op == OP_MALLOC:
            ht[mempos] = OP_MALLOC
            block_amount += 1
        elif op == OP_CALLOC:
            ht[mempos] = OP_CALLOC
            block_amount += 1
        elif op == OP_REALLOC:
            if memptr == '(nil)':
                ht[mempos] = OP_MALLOC
                block_amount += 1
            else:
                if memptr in ht:
                    if ht[memptr] == OP_FREE:
                        continue
                else:
                    continue

                ht[memptr] = OP_FREE
                block_amount -= 1
            if size > 0:
                ht[mempos] = OP_REALLOC
                block_amount += 1

        block_count.append(block_amount)
        time_since_start.append(ts)

    s = pd.Series(block_count, index=time_since_start)
    s.drop_duplicates(inplace=True)
    return {app:s}

def overlap_profile_plot(apps):

    with mp.Pool(processes=len(apps)) as pool:
        dfs = pool.map(aux_overlap_func, apps)

    dfs = {k: v for d in dfs for k, v in d.items()}
    fdf = pd.DataFrame(dfs).fillna(method='ffill')
    fdf.index /= 1E9
    ax = fdf.plot(grid=True, label="Teste paçoca", kind='line')
    ax.set_ylabel('Active memory blocks')
    ax.set_xlabel('Time elapsed in seconds')
    ax.figure.savefig('/tmp/test.pdf')

if __name__ == '__main__':
    pass
    #boxplot(all_apps)
    #plot_corr(all_apps)
    #anova_tukey(all_apps)
    #top_sizes_share_graph(all_apps)
    retention_time_graph(all_apps)
    #problem_graph(all_apps)
    #diff_size_plot(all_apps) # !
    #operation_count_graph(all_apps) # !!
    #pbm_share_group(all_apps) # !!!
    #pbm_share_grouppbm_count(all_apps)
    #overlap_profile_plot(video_player)
