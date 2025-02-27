# Train rest of the TU graph datasets
if [ ! -d "data/pretrained_models" ]; then
    mkdir -p data/pretrained_models
fi

if [ ! -f "data/pretrained_models/gpse_zinc.pt" ]; then
    wget https://zenodo.org/record/8145095/files/gpse_model_zinc_1.0.pt -O data/pretrained_models/gpse_zinc.pt
fi
if [ ! -f "data/pretrained_models/gpse_geom.pt" ]; then
    wget https://zenodo.org/record/8145095/files/gpse_model_geom_1.0.pt -O data/pretrained_models/gpse_geom.pt
fi

if [ ! -f "data/pretrained_models/gpse_molpcba.pt" ]; then
    wget https://zenodo.org/record/8145095/files/gpse_model_molpcba_1.0.pt -O data/pretrained_models/gpse_molpcba.pt
fi

if [ ! -f "data/pretrained_models/gpse_pcqm4mv2.pt" ]; then
    wget https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt -O data/pretrained_models/gpse_pcqm4mv2.pt
fi