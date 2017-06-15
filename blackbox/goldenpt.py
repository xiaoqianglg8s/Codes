from scipy import stats
import numpy as np
import SymbolPd
import TLineGolden

def find_golden_point_ex(x_org, y_org, show=False):
    sp382 = stats.scoreatpercentile(y_org, 38.2)
    sp618 = stats.scoreatpercentile(y_org, 61.8)
    sp50 = stats.scoreatpercentile(y_org, 50.0)

    if show:
        x_org = np.array(x_org)
        sp382_list = [[x_org.min(), x_org.max()], [sp382, sp382]]
        sp618_list = [[x_org.min(), x_org.max()], [sp618, sp618]]
        sp50_list = [[x_org.min(), x_org.max()], [sp50, sp50]]

        TLineDrawer.plot_xy_with_other_x_y(x_org, y_org, '-', sp382_list, sp50_list, sp618_list)
    return sp382, sp50, sp618


def find_golden_point(x_org, y_org, show=False):
    cs_max = y_org.max()
    cs_min = y_org.min()

    sp382 = (cs_max - cs_min) * 0.382 + cs_min
    sp618 = (cs_max - cs_min) * 0.618 + cs_min
    sp50 = (cs_max - cs_min) * 0.5 + cs_min
    if show:
        x_org = np.array(x_org)
        sp382_list = [[x_org.min(), x_org.max()], [sp382, sp382]]
        sp618_list = [[x_org.min(), x_org.max()], [sp618, sp618]]
        sp50_list = [[x_org.min(), x_org.max()], [sp50, sp50]]

        TLineDrawer.plot_xy_with_other_x_y(x_org, y_org, '-', sp382_list, sp50_list, sp618_list)
    return sp382, sp50, sp618


def calc_golden(kl_pd, show=True, only_be=False):
    dk = True if kl_pd.columns.tolist().count('close') > 0 else False
    uq_close = kl_pd.close if dk else kl_pd.price

    if not hasattr(kl_pd, 'name'):
        kl_pd.name = 'unknown'

    g_382, g_500, g_618 = TLineAnalyse.find_golden_point(kl_pd.index, uq_close)
    if show and not only_be:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)
        plt.axhline(g_618, color='c')
        plt.axhline(g_500, color='r')
        plt.axhline(g_382, color='g')
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'g618', 'g500', 'g382'])
        plt.title('mean golden')
        plt.show()

    gex_382, gex_500, gex_618 = TLineAnalyse.find_golden_point_ex(kl_pd.index, uq_close)
    if show and not only_be:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)
        plt.axhline(gex_618, color='c')
        plt.axhline(gex_500, color='r')
        plt.axhline(gex_382, color='g')
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'gex618', 'gex500', 'gex382'])
        plt.title('median golden')
        plt.show()

    above618 = np.maximum(g_618, gex_618)
    below618 = np.minimum(g_618, gex_618)
    above382 = np.maximum(g_382, gex_382)
    below382 = np.minimum(g_382, gex_382)

    percents = [0.20, 0.25, 0.30, 0.70, 0.80, 0.90, 0.95]
    # precents = np.linspace(0.0, 1.0, 0.05)
    pts_dict = TLineAnalyse.find_percent_point(percents, uq_close)

    # import pdb
    # pdb.set_trace()
    below200 = np.minimum(*pts_dict[0.20])
    below250 = np.minimum(*pts_dict[0.25])
    below300 = np.minimum(*pts_dict[0.30])

    above700 = np.maximum(*pts_dict[0.70])
    above800 = np.maximum(*pts_dict[0.80])
    above900 = np.maximum(*pts_dict[0.90])
    above950 = np.maximum(*pts_dict[0.95])

    if show:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)

        plt.axhline(above950, lw=3.5, color='c')
        plt.axhline(above900, lw=3.0, color='y')
        plt.axhline(above800, lw=2.5, color='k')
        plt.axhline(above700, lw=2.5, color='m')

        plt.axhline(above618, lw=2, color='r')
        plt.axhline(below618, lw=1.5, color='r')
        plt.fill_between(kl_pd.index, above618, below618,
                         alpha=0.1, color="r")

        '''
            *************I AM HERE*************
        '''
        plt.axhline(above382, lw=1.5, color='g')
        plt.axhline(below382, lw=2, color='g')
        plt.fill_between(kl_pd.index, above382, below382,
                         alpha=0.1, color="g")

        plt.axhline(below300, lw=2.5, color='k')
        plt.axhline(below250, lw=3.0, color='y')
        plt.axhline(below200, lw=3.5, color='c')

        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'above950', 'above900', 'above800', 'above700', 'above618', 'below618',
                    'above382', 'below382', 'below300', 'below250', 'below200'], bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        plt.title('between golden')
        plt.show()

    return namedtuple('golden', ['g382', 'gex382', 'g500', 'gex500', 'g618',
                                 'gex618', 'above618', 'below618', 'above382', 'below382',
                                 'above950', 'above900', 'above800', 'above700', 'below300', 'below250', 'below200'])(
        g_382, gex_382,
        g_500, gex_500, g_618, gex_618, above618, below618, above382, below382,
        above950, above900, above800, above700, below300, below250, below200)