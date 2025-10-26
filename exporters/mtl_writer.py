from __future__ import annotations
from pathlib import Path


def write_new_mtl(mtl_out_path: str | Path, texture_name: str, material_name: str = "Radioactivity") -> None:
    """
    Écrit un MTL minimal avec un matériau 'material_name' et une texture diffuse 'texture_name'.

    Paramètres
    ----------
    mtl_out_path : str | Path
        Chemin du fichier .mtl à créer.
    texture_name : str
        Nom (ou chemin relatif) du fichier texture (PNG/JPG).
    material_name : str
        Nom du matériau à déclarer dans le MTL.
    """
    p = Path(mtl_out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated MTL for radioactivity mapping\n")
        f.write(f"newmtl {material_name}\n")
        f.write("Kd 1.000 1.000 1.000\n")       # couleur diffuse (blanc)
        f.write("Ka 0.000 0.000 0.000\n")       # ambiante
        f.write("Ks 0.000 0.000 0.000\n")       # spéculaire
        f.write("Ns 1.0\n")                     # brillance
        f.write(f"map_Kd {texture_name}\n")     # texture diffuse
        # Optionnel: f.write("map_d alpha.png\n") pour transparence si besoin


def update_obj_usemtl(
    obj_path_out: str | Path,
    mtllib_name: str,
    usemtl_name: str = "Radioactivity"
) -> None:
    """
    Modifie/insère dans l'OBJ les lignes 'mtllib' et 'usemtl' pour pointer vers le MTL fourni.

    Paramètres
    ----------
    obj_path_out : str | Path
        Chemin du fichier OBJ à modifier (déjà écrit).
    mtllib_name : str
        Nom du fichier MTL (chemin relatif depuis l'OBJ).
    usemtl_name : str
        Nom du matériau à activer dans l'OBJ (doit exister dans le MTL).
    """
    obj_path_out = Path(obj_path_out)
    if not obj_path_out.exists():
        raise FileNotFoundError(f"OBJ introuvable: {obj_path_out}")

    lines = obj_path_out.read_text(encoding="utf-8").splitlines()

    new_lines: list[str] = []
    had_mtllib = False
    had_usemtl = False

    for line in lines:
        if line.startswith("mtllib "):
            if not had_mtllib:
                new_lines.append(f"mtllib {mtllib_name}")
                had_mtllib = True
            # On remplace la première occurrence et on ignore les suivantes
            continue
        if line.startswith("usemtl "):
            if not had_usemtl:
                new_lines.append(f"usemtl {usemtl_name}")
                had_usemtl = True
            continue
        new_lines.append(line)

    # Si mtllib absent → l'insérer au début
    if not had_mtllib:
        new_lines.insert(0, f"mtllib {mtllib_name}")
    # Si usemtl absent → l'insérer après mtllib
    if not had_usemtl:
        insert_idx = 1 if new_lines and new_lines[0].startswith("mtllib ") else 0
        new_lines.insert(insert_idx + 1, f"usemtl {usemtl_name}")

    obj_path_out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
