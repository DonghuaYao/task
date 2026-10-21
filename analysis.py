import pandas as pd

LOSS, COST = 120, 15
raw = pd.read_csv('data/orders.csv', parse_dates=['created_at', 'stock_synced_at'])
print('原始行数', len(raw), '完全重复行', raw.duplicated().sum())
d = raw.drop_duplicates().copy()
print('去重后', len(d), '标签缺失', d.is_fulfilled.isna().sum())
d['lag'] = (d.created_at - d.stock_synced_at).dt.total_seconds() / 60  # 库存数据"过期"多少分钟

# 缺失标签是否集中在某类订单
d['missing'] = d.is_fulfilled.isna()
print('\n缺失率 按lag分段'); print(d.groupby(pd.cut(d.lag, [-1, 15, 30, 61])).missing.mean().round(3))

k = d.dropna(subset=['is_fulfilled']).copy()
k['fail'] = 1 - k.is_fulfilled
print('\n可用订单', len(k), '缺货退单', int(k.fail.sum()), '退单率', round(k.fail.mean(), 4))
print('当前损失(元)', int(k.fail.sum() * LOSS))

print('\n退单率 按lag分段')
print(k.groupby(pd.cut(k.lag, [-1, 5, 10, 15, 20, 25, 30, 45, 61])).fail.agg(['mean', 'sum', 'count']).round(3))
k['lag15'] = k.lag > 15
print('\nlag>15 的退单占全部退单比例', round(k[k.lag15].fail.sum() / k.fail.sum(), 3))
print('\nlag<=15 里 库存低(<=5) vs 高 的退单率')
print(k[~k.lag15].groupby(k.stock_level <= 5).fail.agg(['mean', 'count']).round(3))
print('\nlag>15 里 库存低(<=5) vs 高 的退单率')
print(k[k.lag15].groupby(k.stock_level <= 5).fail.agg(['mean', 'count']).round(3))
print('\n门店/类目 退单率 (lag>15 占比)')
print(k.groupby('store_name').agg(退单率=('fail', 'mean'), 久未同步占比=('lag15', 'mean')).round(3))
print('\n制作耗时分段 退单率'); print(k.groupby(pd.cut(k.prep_time_min, [0, 30, 45, 60, 100])).fail.agg(['mean', 'count']).round(3))

# 规则回放：命中规则的订单提前截单(花15元)，其中本来会退单的省下120元
def replay(mask, name):
    n = int(mask.sum()); hit = int(k.fail[mask].sum())
    net = hit * LOSS - n * COST
    print(f'{name}: 截单{n}单 命中退单{hit} 精确率{hit/n:.1%} 召回{hit/k.fail.sum():.1%} 净收益{net}元')
    return net

print('\n规则回放（阈值扫描）')
for t in [10, 15, 20, 25, 30]:
    replay(k.lag > t, f'lag>{t}')
print()
for t in [15, 20]:
    for s in [5, 10]:
        replay((k.lag > t) & (k.stock_level <= s), f'lag>{t} 且 库存<={s}')
        replay((k.lag > t) | (k.stock_level <= 2), f'lag>{t} 或 库存<=2')

# 防止在同一份数据上挑阈值: 按时间前后各半, 前半选阈值, 后半验证
k = k.sort_values('created_at'); h = len(k) // 2
a, b = k.iloc[:h], k.iloc[h:]
best = max([10, 15, 20, 25, 30], key=lambda t: ((a.lag > t) * a.fail).sum() * LOSS - (a.lag > t).sum() * COST)
m = b.lag > best
print(f'\n前半期选出阈值 lag>{best}，后半期验证: 截单{int(m.sum())} 命中{int(b.fail[m].sum())} 净收益{int(b.fail[m].sum()*LOSS - m.sum()*COST)}元 (后半期原损失{int(b.fail.sum()*LOSS)}元)')

print('\n阈值精细扫描（找断崖位置）')
for t in [6, 8, 9, 10, 11, 12]:
    replay(k.lag > t, f'lag>{t}')
