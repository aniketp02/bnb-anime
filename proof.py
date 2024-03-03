import os
import json
import ezkl

model_path = os.path.join('network.onnx')
compiled_model_path = os.path.join('network.ezkl')
pk_path = os.path.join('test.pk')
vk_path = os.path.join('test.vk')
settings_path = os.path.join('settings.json')

witness_path = os.path.join('witness.json')
data_path = os.path.join('input.json')
cal_data_path = os.path.join('cal_data.json')

# Generate the Witness for the proof

# now generate the witness file
witness_path = os.path.join('witness.json')

res = ezkl.gen_witness(data_path, compiled_model_path, witness_path)
assert os.path.isfile(witness_path)
print('\n\n Witness Generated Successfully!! \n\n')

# Generate the proof

proof_path = os.path.join('proof.json')

proof = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        "single",
    )

print(proof)
print('\n\n')
assert os.path.isfile(proof_path)

# verify our proof

res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
    )

assert res == True
print("\n\nverified")