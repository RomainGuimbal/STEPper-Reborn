import bpy


# bpy.ops.wm.stl_import(filepath="C:/Users/romai/Documents/Ressources/3D assets/3D Prints/surf.stl")

# Simple :

bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/disc in milimeter.step", lin_deflection_bl_unit = 0.5, ang_deflection = 1)

bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/disc in meter.step", lin_deflection_bl_unit = 0.5, ang_deflection = 1)

bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/disc in inches.step", lin_deflection_bl_unit = 0.5, ang_deflection = 1)


# Hierarchy :

#bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/Assy meca standard.stp", lin_deflection_bl_unit = 0.5, ang_deflection = 1)

#bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/Assy meca standard NO CARD-inch.stp", lin_deflection_bl_unit = 0.5, ang_deflection = 1, hierarchy_type="0")

#bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/Assy meca standard NO CARD-inch.stp", lin_deflection_bl_unit = 0.5, ang_deflection = 1, hierarchy_type="1")

bpy.ops.wm.occ_import_step(override_file="C:/Users/romai/Documents/Projets/26 - Bezier Quest/STEP samples/Assy meca standard.stp", lin_deflection_bl_unit = 0.5, ang_deflection = 1)
