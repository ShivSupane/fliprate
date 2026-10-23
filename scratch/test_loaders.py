import tempfile, pathlib, json
import fliprate as fr

# Dakshina
d_dir = tempfile.mkdtemp()
(pathlib.Path(d_dir) / "hi" / "lexicons").mkdir(parents=True, exist_ok=True)
(pathlib.Path(d_dir) / "hi" / "lexicons" / "hi.translit.sampled.train.tsv").write_text("नमस्ते\tnamaste\t10\nनमस्ते\tnamastey\t8\n", encoding="utf-8")
dak_grps = fr.load_dakshina_groups("hi", dakshina_path=d_dir)
dak_rules = fr.dakshina_rules("hi", dakshina_path=d_dir)
print("Dakshina groups:", dak_grps)
print("Dakshina rules count:", len(dak_rules.groups))

# Aksharantar
a_dir = tempfile.mkdtemp()
(pathlib.Path(a_dir) / "hin").mkdir(parents=True, exist_ok=True)
lines = [
    json.dumps({"native word": "नमस्ते", "english word": "namaste"}),
    json.dumps({"native word": "नमस्ते", "english word": "namastey"}),
]
(pathlib.Path(a_dir) / "hin" / "hin_train.json").write_text("\n".join(lines), encoding="utf-8")
aksh_grps = fr.load_aksharantar_groups("hin", aksharantar_path=a_dir)
aksh_rules = fr.aksharantar_rules("hin", aksharantar_path=a_dir)
print("Aksharantar groups:", aksh_grps)
print("Aksharantar rules count:", len(aksh_rules.groups))
print("BOTH LOADED SUCCESSFULLY!")
