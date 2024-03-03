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

# res = ezkl.gen_settings(model_path, settings_path)
# assert res == True

# exit()

# res = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources", max_logrows = 17, scales = [7])
res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
assert res == True

res = ezkl.get_srs( settings_path)

res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
    )


assert res == True
assert os.path.isfile(vk_path)
assert os.path.isfile(pk_path)
assert os.path.isfile(settings_path)