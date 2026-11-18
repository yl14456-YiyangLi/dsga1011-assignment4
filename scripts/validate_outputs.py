"""Validate the saved submission and error-analysis artifacts (CPU, stdlib only)."""
from pathlib import Path
import collections, io, json, pickle

root = Path(__file__).resolve().parents[1]
class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Only primitive record containers are expected')

expected = {'out_original.txt':0.92896, 'out_transformed.txt':0.87808,
    'out_augmented_original.txt':0.92912, 'out_augmented_transformed.txt':0.89820}
for name, accuracy in expected.items():
    values = [int(x) for x in (root/'predictions'/name).read_text().splitlines()]
    assert len(values)==50000 and set(values)<={0,1}
    actual = sum(a==b for a,b in zip(values[::2],values[1::2]))/25000
    assert abs(actual-accuracy)<1e-10
    print(name, f'accuracy={actual:.5f}')
for stem in ['t5_ft_experiment_test','t5_ft_experiment_ec_test']:
    queries = (root/'predictions'/(stem+'.sql')).read_text().splitlines()
    records, errors = PrimitiveUnpickler(io.BytesIO((root/'predictions'/(stem+'.pkl')).read_bytes())).load()
    assert len(queries)==len(records)==len(errors)==432
    assert all(q.strip() for q in queries)
    print(stem, 'queries=432', 'execution_errors=', sum(bool(e) for e in errors))
summary=json.loads((root/'analysis/error_summary.json').read_text())
for model in ['finetuned','scratch']:
    examples=json.loads((root/'analysis'/(model+'_dev_errors.json')).read_text())
    assert len(examples)==466
    for category,count in collections.Counter(x['error_category'] for x in examples).items():
        assert summary[model][category]==count
    assert sum(not x['record_exact_match'] for x in examples)==summary[model]['record_mismatches']
print('PASS: saved prediction files and error-analysis counts')
