# Preset: toy (numpy policy with untrained padded inputs; whole lifecycle in seconds)

    python "$(python -c "import editstack,os;print(os.path.dirname(editstack.__file__))")/presets/toy/make_model.py" OUT
    editstack new ./toy-stack --weights OUT --preset toy --task "reach a 2-D target"
