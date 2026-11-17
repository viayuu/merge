"""Fine-tune one task from the shared URS checkpoint using its original RL trainer."""

import argparse
import atexit
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
URS = ROOT / 'URS'
sys.path.insert(0, str(URS))

from Trainer import Trainer
from utils.utils import AverageMeter, create_logger, seed_everything


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FineTuneTrainer(Trainer):
    def _train_one_epoch(self, epoch):
        # 保留原来的单批 rollout 和 REINFORCE 更新，只按实例数控制 epoch。
        task = self.train_problem_list[0]
        target = self.trainer_params['instances_per_epoch']
        batch_size = self.trainer_params['batch_size']
        loss = AverageMeter()
        seen = 0
        batch = 0
        started = time.monotonic()
        while seen < target:
            current_batch = min(batch_size, target - seen)
            score, batch_loss = self._train_one_batch(
                task, self.env_params['problem_size'], current_batch,
                {'capacity': self.env_params['capacity']},
            )
            loss.update(batch_loss, current_batch)
            seen += current_batch
            batch += 1
            if batch <= 5 or batch % 100 == 0 or seen == target:
                self.logger.info(
                    'Epoch %d: instances %d/%d, batch %d, score %.6f, loss %.6f, elapsed %.1fs',
                    epoch, seen, target, batch, score, loss.avg, time.monotonic() - started,
                )
        return loss.avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-dir', '--out-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=URS / 'pretrained/unified_checkpoint_500.pt')
    parser.add_argument('--data-dir', type=Path, default=URS / 'dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--instances-per-epoch', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr-decay-epoch', type=int, default=26)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--validation-episodes', type=int, default=1000)
    parser.add_argument('--validation-batch-size', type=int, default=125)
    args = parser.parse_args()
    args.base = args.base.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 同一专家可被提前分配到另一节点；原队列稍后只等待并复用其完整结果。
    # 共享盘的 flock 不跨节点互斥，因此使用已验证的原子 mkdir。
    training_lock = args.output_dir / '.training_claim'
    while True:
        try:
            training_lock.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    atexit.register(training_lock.rmdir)
    metadata_path = args.output_dir / 'finetune_metadata.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous['status'] == 'completed':
            assert previous['task'] == args.task and previous['seed'] == args.seed
            assert previous['base_file_sha256'] == hashlib.sha256(args.base.read_bytes()).hexdigest()
            for key in ['epochs', 'instances_per_epoch', 'batch_size']:
                assert previous['trainer_params'][key] == getattr(args, key)
            assert previous['env_params']['data_dir'] == str(args.data_dir)
            assert previous['optimizer_params']['optimizer']['lr'] == args.lr
            assert previous['optimizer_params']['lr_decay_epoch'] == [args.lr_decay_epoch]
            assert Path(previous['final_checkpoint']).is_file()
            print(f'Reusing completed {args.task} fine-tune: {previous["final_checkpoint"]}', flush=True)
            return
        raise RuntimeError(f'Existing incomplete fine-tune: {metadata_path}; inspect before restarting')
    if args.cuda >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this run but is unavailable')

    seed_everything(args.seed)
    create_logger(log_file={'filepath': str(args.output_dir), 'filename': 'run.log'})
    trainer = FineTuneTrainer(
        env_params={'problem_size': 100, 'capacity': 50, 'data_dir': str(args.data_dir)},
        model_params={
            'embedding_dim': 128, 'encoder_layer_num': 12, 'ff_hidden_dim': 512,
            'logit_clipping': 50, 'demand_max1': True, 'eval_type': 'greedy',
        },
        optimizer_params={
            'optimizer_type': 'AdamW',
            'optimizer': {'lr': args.lr, 'weight_decay': 1e-6},
            'lr_decay_epoch': [args.lr_decay_epoch],
        },
        trainer_params={
            'use_cuda': args.cuda >= 0, 'cuda_device_num': args.cuda,
            'epochs': args.epochs, 'instances_per_epoch': args.instances_per_epoch,
            'batches_per_epoch': math.ceil(args.instances_per_epoch / args.batch_size),
            'batch_size': args.batch_size, 'train_problem_list': [args.task],
            'validation_problem_list': [args.task], 'validation_scale': 100,
            'validation_episodes': args.validation_episodes,
            'validation_batch_size': args.validation_batch_size,
            'model_load': {'enable': False},
            'logging': {
                'model_save_interval': 1,
                'log_image_params_1': {'json_foldername': 'log_image_style', 'filename': 'style_score.json'},
                'log_image_params_2': {'json_foldername': 'log_image_style', 'filename': 'style_loss.json'},
            },
        },
    )

    # 只加载共同 base 的权重，保留新优化器、从 1 开始的 epoch 和独立日志。
    with torch.serialization.safe_globals([set]):
        checkpoint = torch.load(args.base, map_location='cpu', weights_only=True)
    base_state = checkpoint['model_state_dict']
    trainer.model.load_state_dict(base_state, strict=True)
    base_hash = state_hash(base_state)
    initial_hash = state_hash(trainer.model.state_dict())
    assert initial_hash == base_hash and not trainer.optimizer.state
    metadata = {
        'task': args.task, 'base_checkpoint': str(args.base), 'base_epoch': checkpoint['epoch'],
        'base_file_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
        'base_model_sha256': base_hash, 'initial_model_sha256': initial_hash,
        'optimizer_state_reset': True, 'start_epoch': trainer.start_epoch,
        'env_params': trainer.env_params, 'model_params': trainer.model_params,
        'optimizer_params': trainer.optimizer_params, 'trainer_params': trainer.trainer_params,
        'seed': args.seed, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(args.cuda) if args.cuda >= 0 else 'cpu',
        'validation_protocol': 'Original Trainer: own task, greedy POMO, no augmentation',
        'final_checkpoint': str(args.output_dir / f'checkpoint_{args.epochs}.pt'),
        'status': 'running',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    trainer.logger.info('Fine-tune configuration: %s', json.dumps(metadata))
    del checkpoint, base_state
    started = time.monotonic()
    trainer.run()
    metadata.update(status='completed', elapsed_seconds=time.monotonic() - started)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
