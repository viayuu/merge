"""Continue the original URS model and optimizer on five tasks with fixed budgets."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
URS = ROOT / 'URS'
sys.path.insert(0, str(URS))

from Trainer import Trainer
from utils.utils import AverageMeter, create_logger, seed_everything


TASKS = ['cvrp', 'cvrpb', 'cvrptw', 'ocvrp', 'ocvrptw']


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


class ContinueTrainer(Trainer):
    def _train_one_epoch(self, epoch):
        # Keep the original batch-level task sampling and REINFORCE update.
        # Only the stopping condition and final partial batches differ.
        quota = self.trainer_params['instances']
        per_task = self.trainer_params['budget'] == 'per-task'
        target = quota * len(self.train_problem_list) if per_task else quota
        batch_size = self.trainer_params['batch_size']
        task_instances = dict.fromkeys(self.train_problem_list, 0)
        task_batches = dict.fromkeys(self.train_problem_list, 0)
        seen = 0
        batches = 0
        loss = AverageMeter()
        started = time.monotonic()

        while seen < target:
            available = ([task for task in self.train_problem_list
                          if task_instances[task] < quota]
                         if per_task else self.train_problem_list)
            task = random.choice(available)
            remaining = quota - task_instances[task] if per_task else target - seen
            current_batch = min(batch_size, remaining)
            score, batch_loss = self._train_one_batch(
                task, self.env_params['problem_size'], current_batch,
                {'capacity': self.env_params['capacity']},
            )
            loss.update(batch_loss, current_batch)
            seen += current_batch
            batches += 1
            task_instances[task] += current_batch
            task_batches[task] += 1
            if batches <= 5 or batches % 100 == 0 or seen == target:
                self.logger.info(
                    'Epoch %d: instances %d/%d, batch %d/%d, task %s, '
                    'score %.6f, loss %.6f, elapsed %.1fs',
                    epoch, seen, target, batches,
                    self.trainer_params['batches_per_epoch'], task, score,
                    loss.avg, time.monotonic() - started,
                )
                progress = {
                    'absolute_epoch': epoch,
                    'additional_epoch': epoch - self.start_epoch + 1,
                    'completed_instances': seen, 'target_instances': target,
                    'completed_batches': batches,
                    'target_batches': self.trainer_params['batches_per_epoch'],
                    'task_instances': task_instances, 'task_batches': task_batches,
                    'loss': loss.avg, 'lr': self.optimizer.param_groups[0]['lr'],
                    'elapsed_seconds': time.monotonic() - started,
                    'phase': 'training_epoch_complete' if seen == target else 'training',
                }
                write_json(Path(self.result_folder) / 'progress.json', progress)

        with (Path(self.result_folder) / 'training_counts.jsonl').open('a') as handle:
            handle.write(json.dumps(progress) + '\n')
        return loss.avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', required=True, choices=['total', 'per-task'])
    parser.add_argument('--instances', type=int, default=100000,
                        help='Instances per epoch, either total or for each task')
    parser.add_argument('--epochs', type=int, default=30, help='Additional epochs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base', type=Path,
                        default=URS / 'pretrained/unified_checkpoint_500.pt')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--validation-episodes', type=int, default=1000)
    parser.add_argument('--validation-batch-size', type=int, default=1000)
    args = parser.parse_args()
    if args.instances <= 0 or args.epochs <= 0:
        parser.error('--instances and --epochs must be positive')
    args.base = args.base.resolve()
    args.output_dir = args.output_dir.resolve()
    metadata_path = args.output_dir / 'metadata.json'
    if metadata_path.exists():
        raise RuntimeError(f'Existing run: {metadata_path}; use a new output directory')
    if args.cuda >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.serialization.safe_globals([set]):
        checkpoint = torch.load(args.base, map_location='cpu', weights_only=True)
    base_epoch = checkpoint['epoch']
    final_epoch = base_epoch + args.epochs
    multiplier = len(TASKS) if args.budget == 'per-task' else 1
    batches_per_epoch = math.ceil(args.instances / 128) * multiplier

    seed_everything(args.seed)
    create_logger(log_file={'filepath': str(args.output_dir), 'filename': 'run.log'})
    trainer = ContinueTrainer(
        env_params={'problem_size': 100, 'capacity': 50, 'data_dir': str(URS / 'dataset')},
        model_params={
            'embedding_dim': 128, 'encoder_layer_num': 12, 'ff_hidden_dim': 512,
            'logit_clipping': 50, 'demand_max1': True, 'eval_type': 'greedy',
        },
        optimizer_params={
            'optimizer_type': 'AdamW',
            'optimizer': {'lr': 1e-4, 'weight_decay': 1e-6},
            'lr_decay_epoch': [451],
        },
        trainer_params={
            'use_cuda': args.cuda >= 0, 'cuda_device_num': args.cuda,
            'epochs': final_epoch, 'budget': args.budget, 'instances': args.instances,
            'batches_per_epoch': batches_per_epoch, 'batch_size': 128,
            'train_problem_list': TASKS, 'validation_problem_list': TASKS,
            'validation_scale': 100, 'validation_episodes': args.validation_episodes,
            'validation_batch_size': args.validation_batch_size,
            'model_load': {'enable': False},
            'logging': {
                'model_save_interval': 1,
                'log_image_params_1': {'json_foldername': 'log_image_style',
                                       'filename': 'style_score.json'},
                'log_image_params_2': {'json_foldername': 'log_image_style',
                                       'filename': 'style_loss.json'},
            },
        },
    )

    # Match Trainer's resume semantics while accepting the released filename.
    trainer.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    trainer.result_log.set_raw_data(checkpoint['result_log'])
    trainer.start_epoch = base_epoch + 1
    base_hash = state_hash(checkpoint['model_state_dict'])
    initial_hash = state_hash(trainer.model.state_dict())
    assert initial_hash == base_hash
    initial_steps = Counter(int(state['step'].item())
                            for state in trainer.optimizer.state.values())
    group = trainer.optimizer.param_groups[0]
    metadata = {
        'budget': args.budget, 'instances': args.instances,
        'instances_per_epoch_total': args.instances * multiplier,
        'additional_epochs': args.epochs,
        'base_checkpoint': str(args.base), 'base_epoch': base_epoch,
        'base_file_sha256': hashlib.sha256(args.base.read_bytes()).hexdigest(),
        'base_model_sha256': base_hash, 'initial_model_sha256': initial_hash,
        'start_epoch': trainer.start_epoch, 'final_epoch': final_epoch,
        'optimizer_state_restored': True, 'optimizer_state_reset': False,
        'initial_lr': group['lr'], 'initial_optimizer_steps': dict(initial_steps),
        'optimizer_state_count': len(trainer.optimizer.state),
        'optimizer_type': 'AdamW', 'weight_decay': group['weight_decay'],
        'betas': group['betas'], 'eps': group['eps'], 'lr_decay_epoch': [451],
        'env_params': trainer.env_params, 'model_params': trainer.model_params,
        'trainer_params': trainer.trainer_params,
        'seed': args.seed, 'rng_state_restored': False,
        'rng_note': 'The released checkpoint has no RNG state; seed starts a new sequence.',
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(args.cuda) if args.cuda >= 0 else 'cpu',
        'validation_protocol': 'Original Trainer: five tasks, greedy POMO, no augmentation',
        'final_checkpoint': str(args.output_dir / f'checkpoint_{final_epoch}.pt'),
        'status': 'running',
    }
    write_json(metadata_path, metadata)
    trainer.logger.info('Continuation configuration: %s', json.dumps(metadata))
    del checkpoint
    started = time.monotonic()
    trainer.run()
    elapsed = time.monotonic() - started
    metadata.update(status='completed', elapsed_seconds=elapsed)
    write_json(metadata_path, metadata)
    progress_path = args.output_dir / 'progress.json'
    progress = json.loads(progress_path.read_text())
    progress.update(phase='completed', training_elapsed_seconds=elapsed)
    write_json(progress_path, progress)


if __name__ == '__main__':
    main()
