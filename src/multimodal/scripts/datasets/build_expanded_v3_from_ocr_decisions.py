"""Build expanded_v3 eval dataset from OCR/manual duplicate decisions.

This writes evaluation artifacts only and never modifies business image bindings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open('r', encoding='utf-8') if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows), encoding='utf-8')


def build(dataset: str, decisions: str, output: str, dataset_version: str, report_dir: str) -> dict[str, Any]:
    rows = [dict(row) for row in read_jsonl(dataset)]
    by_asset = {str(row.get('asset_id')): row for row in rows}
    decision_rows = read_jsonl(decisions)
    decision_by_asset = {str(row.get('asset_id')): row for row in decision_rows}

    restored_assets: set[str] = set()
    excluded_assets: set[str] = set()
    shared_assets: set[str] = set()
    needs_rebind_assets: set[str] = set()
    missing_decision_assets = [asset_id for asset_id in decision_by_asset if asset_id not in by_asset]

    for item in rows:
        asset_id = str(item.get('asset_id'))
        item['dataset_version'] = dataset_version
        metadata = dict(item.get('metadata') or {})
        metadata['expanded_v3_source_dataset'] = dataset
        decision = decision_by_asset.get(asset_id)
        if decision:
            metadata['expanded_v3_decision_source'] = decision.get('decision_source')
            metadata['expanded_v3_decision'] = decision.get('decision')
            metadata['expanded_v3_eval_action'] = decision.get('eval_action')
            metadata['expanded_v3_reason'] = decision.get('reason')
            if decision.get('eval_action') == 'keep':
                item['usable'] = True
                item['visual_label_status'] = 'contextual'
                metadata['exclusion_reason'] = None
                metadata['expanded_v3_restored_from_review'] = True
                restored_assets.add(asset_id)
            elif decision.get('eval_action') == 'exclude_shared':
                item['usable'] = False
                item['visual_label_status'] = 'invalid'
                metadata['exclusion_reason'] = 'expanded_v3_shared_image_excluded_from_single_label_eval'
                shared_assets.add(asset_id)
            elif decision.get('eval_action') == 'needs_rebind':
                item['usable'] = False
                item['visual_label_status'] = 'invalid'
                metadata['exclusion_reason'] = 'expanded_v3_wrong_binding_needs_rebind'
                needs_rebind_assets.add(asset_id)
            else:
                item['usable'] = False
                item['visual_label_status'] = 'invalid'
                metadata['exclusion_reason'] = metadata.get('exclusion_reason') or 'expanded_v3_duplicate_ocr_excluded'
                excluded_assets.add(asset_id)
        item['metadata'] = metadata

    usable_by_node: dict[str, set[str]] = defaultdict(set)
    for item in rows:
        if item.get('usable', True) is False:
            continue
        usable_by_node[str(item.get('node_id') or '')].add(str(item.get('role') or ''))
    invalid_nodes = {node_id for node_id, roles in usable_by_node.items() if not {'query', 'gallery'}.issubset(roles)}

    invalidated_after_review = 0
    for item in rows:
        if item.get('usable', True) is False:
            continue
        node_id = str(item.get('node_id') or '')
        if node_id in invalid_nodes:
            metadata = dict(item.get('metadata') or {})
            item['usable'] = False
            item['visual_label_status'] = 'invalid'
            metadata['exclusion_reason'] = 'node_without_query_gallery_after_expanded_v3_review'
            item['metadata'] = metadata
            invalidated_after_review += 1

    write_jsonl(output, rows)
    usable = [row for row in rows if row.get('usable', True) is not False]
    per_node_counts = Counter(str(row.get('node_id')) for row in usable)
    image_count_distribution = Counter(str(v) for v in per_node_counts.values())
    role_counts = Counter(str(row.get('role') or '') for row in usable)
    decision_counts = Counter(str(row.get('eval_action') or row.get('decision') or '') for row in decision_rows)

    summary = {
        'dataset_version': dataset_version,
        'source_dataset': dataset,
        'decisions': decisions,
        'output': output,
        'items': len(rows),
        'usable': len(usable),
        'excluded': len(rows) - len(usable),
        'usable_nodes': len(per_node_counts),
        'query_items': role_counts.get('query', 0),
        'gallery_items': role_counts.get('gallery', 0),
        'image_count_distribution': dict(sorted(image_count_distribution.items(), key=lambda x: int(x[0]))),
        'decision_counts': dict(decision_counts),
        'restored_assets': len(restored_assets),
        'auto_or_manual_excluded_assets': len(excluded_assets),
        'shared_assets_excluded': len(shared_assets),
        'needs_rebind_assets': len(needs_rebind_assets),
        'invalid_nodes_after_review': len(invalid_nodes),
        'invalidated_items_after_review': invalidated_after_review,
        'missing_decision_assets': missing_decision_assets,
    }
    Path(output).with_suffix('.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = [
        '# Expanded v3 Dataset Report',
        '',
        '## Summary',
        '',
        f"- source: `{dataset}`",
        f"- output: `{output}`",
        f"- usable images/nodes: {summary['usable']} / {summary['usable_nodes']}",
        f"- query/gallery: {summary['query_items']} / {summary['gallery_items']}",
        f"- restored by manual OCR review: {summary['restored_assets']}",
        f"- duplicate/shared/excluded decisions: {summary['auto_or_manual_excluded_assets']} / {summary['shared_assets_excluded']}",
        '',
        '## Decision Policy',
        '',
        '- same-node same-URL duplicates: auto exclude duplicate asset, keep the smallest asset_id representative.',
        '- manual confirm: restore into single-label eval as hard negative / valid binding.',
        '- manual shared: exclude from single-label eval, candidate for future multilabel/shared-asset eval.',
        '- manual exclude: keep excluded from eval.',
    ]
    (out_dir / 'EXPANDED_V3_REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build expanded_v3 from OCR duplicate decisions.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--decisions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-version', default='scenic_4_image_node_expanded_v3')
    parser.add_argument('--report-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.decisions, args.output, args.dataset_version, args.report_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
