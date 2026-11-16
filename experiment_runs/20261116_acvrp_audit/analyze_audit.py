"""Verify targeted controls and preserve the ACVRP diagnosis."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'experiment_runs/20261116_four_gpu'
CASES = ['gpu03_standalone', 'gpu05_standalone', 'gpu05_after_atsp',
         'gpu03_rng_advanced', 'gpu05_rng_advanced']
MODELS = {'unified': 'unified__acvrp', 'specialist': 'specialist_acvrp__acvrp'}


def read(path):
    return json.loads(path.read_text())


def main():
    results, traces, csv_rows = {}, {}, []
    for model, previous_id in MODELS.items():
        previous_path = PREVIOUS / 'results' / f'{previous_id}.json'
        previous = read(previous_path)['rows'][0]
        for case in CASES:
            path = RUN / 'results' / f'{model}_{case}.json'
            payload = read(path)
            trace = read(path.with_suffix('.trace.json'))
            row = next(row for row in payload['rows'] if row['task'] == 'acvrp')
            assert not payload['missing'] and payload['seed'] == 3407
            assert payload['model_params']['eval_type'] == 'greedy'
            assert row['scale'] == 100 and row['episodes'] == 1000
            assert row['aug_factor'] == 128 and row['configured_batch_size'] == 50
            assert row['pomo_size'] == 100 and payload['device'] == 'cuda:0'
            assert row['data_path'] == previous['data_path']
            assert row['solution_path'] is None
            assert row['oracle_score'] == previous['oracle_score']
            assert payload['checkpoint'] == read(previous_path)['checkpoint']
            results[(model, case)] = row
            calls = [call for call in trace['calls'] if call['task'] == 'acvrp']
            assert len(calls) == 20
            traces[(model, case)] = calls
            csv_rows.append({
                'model': model, 'case': case, 'gpu': trace['device_name'],
                'task_order': ','.join(row['task'] for row in payload['rows']),
                'gap_pct': row['aug_gap'], 'objective': row['aug_score'],
                'reference': row['oracle_score'], 'source_json': str(path),
            })
        assert results[(model, 'gpu03_standalone')]['aug_gap'] == previous['aug_gap']
        assert results[(model, 'gpu03_standalone')]['aug_score'] == previous['aug_score']
        a, b = traces[(model, 'gpu05_after_atsp')], traces[(model, 'gpu05_rng_advanced')]
        for x, y in zip(a, b):
            assert all(x[key] == y[key] for key in ['rng_before', 'rng_after', 'features_sha256'])
        for key in ['aug_gap', 'aug_score', 'no_aug_gap', 'no_aug_score']:
            assert results[(model, 'gpu05_after_atsp')][key] == results[(model, 'gpu05_rng_advanced')][key]
        assert all(x['features_sha256'] != y['features_sha256'] for x, y in zip(
            traces[(model, 'gpu05_standalone')], a))
        assert all(x['features_sha256'] != y['features_sha256'] for x, y in zip(
            traces[(model, 'gpu03_standalone')], traces[(model, 'gpu05_standalone')]))
    assert f'{results[("unified", "gpu05_after_atsp")]["aug_gap"]:.2f}' == '3.06'
    assert f'{results[("specialist", "gpu05_after_atsp")]["aug_gap"]:.2f}' == '2.01'
    with (RUN / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    proof = {'created': datetime.now().astimezone().isoformat(), 'new_acvrp_runs': 10,
             'all_original_data_and_cli_parameters': True, 'rng_advanced_cases_are_diagnostic': True, 'repeat_3090_exact': True,
             'sequence_vs_rng_only_all_20_batches_exact': True,
             'sequence_vs_rng_only_all_scores_exact': True,
             'paper_two_decimal_match': True, 'hashes': {}}
    for path in [ROOT / 'URS/dataset/acvrp/acvrp100_n1000_seed1234.pt',
                 ROOT / 'URS/pretrained/unified_checkpoint_500.pt',
                 ROOT / 'URS/pretrained/acvrp_checkpoint_300.pt']:
        proof['hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RUN / 'verification.json').write_text(json.dumps(proof, indent=2) + '\n')

    report = ['# ACVRP 为什么与论文相差一点？', '',
        '2026-11-16，Asia/Shanghai。针对性复跑已完成：10个ACVRP评测进程，其中2个含ATSP前置任务；全部成功，原数据、checkpoint及评测参数不变。', '',
        '**原因已定位：把每个任务拆成独立进程，改变了ACVRP随机节点标识所处的随机序列；换卡还会带来较小的差异。4090上ATSP→ACVRP的结果恢复为论文的3.06%/2.01%。**', '',
        '**原表正确吗？**', '',
        '正确。除ACVRP之外，10个任务的本地URS与本地专家结果保留两位小数均与论文吻合。ACVRP本地分别为3.0418372071%和1.9995601920%，保留两位为3.04%/2.00%；与论文3.06%/2.01%的差为−0.0181627929/−0.0104398080个百分点。', '',
        '**原数据复跑得到什么？**', '',
        '| 设置 | GPU | URS gap % | ACVRP专家 gap % |',
        '|---|---|---:|---:|',
        '| 论文 | 4090 | 3.06 | 2.01 |']
    for case, label, gpu in [
        ('gpu03_standalone', '单独ACVRP；与原本地结果逐位相同', '3090'),
        ('gpu05_standalone', '单独ACVRP；只换GPU', '4090'),
        ('gpu05_after_atsp', '同一进程ATSP→ACVRP', '4090'),
        ('gpu05_rng_advanced', '只推进ATSP等量随机数，再跑ACVRP', '4090'),
        ('gpu03_rng_advanced', '只推进ATSP等量随机数，再跑ACVRP', '3090'),
    ]:
        u, s = [results[(model, case)]['aug_gap'] for model in MODELS]
        report.append(f'| {label} | {gpu} | {u:.9f} | {s:.9f} |')
    report += ['',
        '全部使用同一原ACVRP数据1000例、N100、seed3407、greedy、POMO100、增强128、batch50。每次的参考均值完全相同，为2.1027939319610596。没有更换参考解、样本或checkpoint。', '',
        '**为什么greedy仍受随机数影响？**', '',
        f'原[Model_LIB.py]({ROOT}/URS/model/Model_LIB.py:208)对非对称任务使用`torch.rand`生成节点标识；greedy只固定动作选择为argmax，并没有固定模型输入中的随机标识。128份增强副本共享原距离和需求，但有不同随机标识。', '',
        f'原[test.py]({ROOT}/URS/test.py:135)只在进程开始设置一次seed，[Tester.py]({ROOT}/URS/Tester.py:144)按任务顺序继续使用同一CUDA随机数生成器。此前45项队列则为每个模型任务启动独立进程，每次重新seed；因此ACVRP从初始随机状态开始。', '',
        f'原[train_problem_list]({ROOT}/URS/problem/ProblemSet.py:140)按名字长度稳定排序后为OP、TSP、ATSP、CVRP、ACVRP、PCTSP、CVRPB、OCVRP、PDTSP、CVRPTW、OCVRPTW。ACVRP之前只有ATSP使用随机节点标识。ATSP的1000例按batch50执行20批，每次生成(50×128,100,1)的随机标识；随后ACVRP接着这个随机序列生成自己的标识。', '',
        '实测4090上，单独ACVRP与ATSP后ACVRP的20批节点标识摘要全部不同。只在ACVRP开始前消耗20次同形状torch.rand，不执行任何ATSP推理，得到的20批随机状态与节点标识摘要逐批完全等于ATSP→ACVRP；两个模型的增强/非增强objective及gap也逐位一致。这将差异定位到RNG推进，排除前一任务环境或解码器状态残留的解释。', '',
        '**为什么其余任务没出现这个两位小数差异？**', '',
        '对称任务使用固定坐标，不调用上述随机节点标识；ATSP是这11个任务里最先使用随机标识的任务，在4090上本次前置ATSP的URS结果仍为2.2581379294%，与原45项中的ATSP逐位一致。ACVRP是后续非对称任务，因此更容易暴露按任务重设seed与连续评测的区别。', '',
        '**GPU型号有什么影响？**', '',
        '同seed的单独ACVRP从3090换到4090，URS改变−0.003548284个百分点，专家改变+0.002630458个百分点。追踪确认，两种卡在首次特征构造前的CUDA RNG状态摘要相同，但生成的随机节点标识不同；20批摘要均不同。不能仅将其称为浮点运算误差。', '',
        '[PyTorch 2.5.1的CUDA随机数实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/DistributionTemplates.h#L44)按GPU的SM数量及每SM线程数设置生成网格和Philox counter推进量，因此相同seed和形状并不保证3090/4090生成逐元素相同的随机标识。这与本次追踪一致。', '',
        '**对原实验有什么影响？**', '',
        '上一批本地ACVRP数值是真实、可逐位重复的结果。它与本次按4090、ATSP→ACVRP执行所得到的论文精度结果，使用了不同随机标识。此前“沿用seed和batch即能完全对齐论文”的隐含假设不够；复现时还需记录GPU型号和进入任务前的CUDA RNG状态/任务顺序。', '',
        '此前45项中URS、ACVRP专家和11专家平均在ACVRP上均用3090、相同seed、batch50且单任务进程，保持了这三者的输入随机性设置一致；ATSP的对应三者均使用4090。当前诊断不改变已经保存的负面合并结果。原CSV和JSON保留，不覆盖为论文数字。', '',
        '后续若要按论文数字复现，可在4090上保持ATSP→ACVRP的原连续执行；若为了并行拆分任务，应显式记录并恢复相应任务的随机状态，同时固定GPU型号与batch。没有必要修改原数据集或gap。', '',
        '论文没有提供用于表格的完整命令及RNG状态日志，因此不能断言作者历史运行必然采用该命令；这里已经用原代码建立了可重复的差异机制，并在两个checkpoint上同时复现了论文发表精度。', '',
        f'完整证据：[控制实验CSV]({RUN}/comparison.csv)、[机器可读复核]({RUN}/verification.json)、[原始结果目录]({RUN}/results)、[运行说明]({RUN}/README.md)。所有诊断脚本仅位于本运行目录，URS源码未新增修改。']
    (RUN / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print('PASS: 10 original-data ACVRP runs; RNG-only control exactly reproduces ATSP→ACVRP.')
    print(RUN / 'RESULTS.md')


if __name__ == '__main__':
    main()
