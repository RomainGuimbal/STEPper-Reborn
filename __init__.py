# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2021 Tommi Hyppänen
#
## Modified 2025 Romain Guimbal

import ntpath
import os
import time
import math
import bpy

from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from .utils import (
    obj_unlink_all,
    calculate_detail_level,
    transform_to_up,
    choose_hierarchy_types,
)
from .build_mesh import build_mesh, bl_obj_from_occ_shape
from .build_blender_hierarchy import build_blender_hierarchy

global_file_cache = {}


def load_step(
    context,
    filepath,
    custom_scale=None,
    lin_deflection=0.8,
    ang_deflection=0.5,
    # merge_distance=0.001,
    up_as="Y",
    htypes="TREE",
):
    from .step_reader import ReadSTEP

    hierarchy_flat, hierarchy_tree, hierarchy_empties = choose_hierarchy_types(htypes)

    filename = "".join(ntpath.basename(filepath).split(".")[:-1])

    if filepath not in global_file_cache:
        try:
            step_reader = ReadSTEP(filepath)
            global_file_cache[filepath] = step_reader
        except AssertionError as e:
            print(e)
            return False

    else:
        step_reader = global_file_cache[filepath]
        print("Loaded file from cache")

    tree = step_reader.tree
    scale = step_reader.scale
    if custom_scale is not None:
        scale = custom_scale

    # divide by Blender unit length
    scale /= context.scene.unit_settings.scale_length
    print("Current Blender scale set at:", context.scene.unit_settings.scale_length)

    wm = bpy.context.window_manager

    created_objs = []
    created_names = {}
    created_uuid = {}

    # traverse shapes, render in "face" mode
    start_time = time.time()
    all_shapes = tree.get_shapes()
    total = len(all_shapes)

    wm.progress_begin(0, total)
    for i, (shp, node_index) in enumerate(all_shapes):
        obj = bl_obj_from_occ_shape(
            step_reader,
            shp,
            tree,
            filename,
            filepath,
            hierarchy_empties,
            node_index,
            created_names,
            lin_deflection,
            ang_deflection,
            created_uuid,
            total,
            i,
        )
        if obj:
            created_objs.append(obj)
        wm.progress_update(i)

    # assert len(created_objs) == len(shapes_labels)
    print("\n" + repr(step_reader.import_problems))

    # remove all temporary links
    for tobj in created_objs:
        obj_unlink_all(tobj)

    # build hierarchy
    build_blender_hierarchy(
        filename,
        tree,
        created_objs,
        hierarchy_flat,
        hierarchy_tree,
        hierarchy_empties,
        created_uuid,
    )

    transform_to_up(up_as[0], created_objs, scale)

    wm.progress_end()
    print(f"STEP loading time elapsed: {time.time()-start_time:.2f}")

    return True


class PG_Stepper(bpy.types.PropertyGroup):
    build_materials: bpy.props.BoolProperty(
        name="Build Materials",
        description="Build materials from STEP file colors",
        default=True,
    )

    hack_skip_zero_solids: bpy.props.BoolProperty(
        name="Skip Faulty Solids",
        description="Skip some shapes the library hangs on and fails to load",
        default=False,
    )

    simpler_parameters: bpy.props.BoolProperty(
        name="Artist Friendly Parameters",
        description="Instead of linear and angle deflection values, use only detail setting",
        default=False,
    )

    detail_level: bpy.props.IntProperty(
        name="Mesh Detail",
        description="How detailed you want the mesh to be",
        default=100,
        min=1,
    )

    # In meter. Must be multiplied by 2000 to match OCC deflection length.
    lin_deflection: bpy.props.FloatProperty(
        name="Linear Deflection",
        description="Max distance between the mesh and the theoretical shape. Smaller values increase polygon count",
        default=0.001,  # 1mm
        min=0.00001,  # 0.01mm
        unit="LENGTH",
        step=0.01,
    )

    # In radian. Must be multiplied by 2 to match OCC deflection angle.
    ang_deflection: bpy.props.FloatProperty(
        name="Angular Deflection",
        description="Max angle between the tangent plane and the surrounding mesh of samples. Smaller values increase polygon count",
        default=0.0872664,  # 5°
        soft_min=0.00174532925,  # 0.1°
        min=0.000001745,  # 0.0001°
        max=math.pi,
        unit="ROTATION",
        step=100,  # 1°
    )

    fix_ascii_file: bpy.props.StringProperty(
        name="File",
        description="Path to problematic STEP file",
        default="",
        maxlen=1024,
        subtype="FILE_PATH",
    )

    use_adaptive_resolution: bpy.props.BoolProperty(
        name="Adaptive resolution",
        description="Automatically adjust deflection values based on shape size",
        default=True,
    )


class STEP_OT_ImportStepCADOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "object.occ_import_step"
    bl_label = "Import STEP"
    bl_description = "Import a STEP file"
    bl_options = {"PRESET"}

    filter_glob: StringProperty(default="*.step;*.stp;*.st", options={"HIDDEN"})
    files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    # files: bpy.props.CollectionProperty(type=idprop.types.IDPropertyGroup)
    override_file: StringProperty(default="", options={"HIDDEN"})

    fw_as: bpy.props.EnumProperty(
        items=[
            ("XPOS", "X", "", 0),
            # ("XNEG", "X-", "", 1),
            ("YPOS", "Y", "", 2),
            # ("YNEG", "Y-", "", 3),
            ("ZPOS", "Z", "", 4),
            # ("ZNEG", "Z-", "", 5),
        ],
        name="Forward",
        default="ZPOS",
        description="Forward axis of the imported model",
    )

    up_as: bpy.props.EnumProperty(
        items=[
            ("XPOS", "X", "", 0),
            # ("XNEG", "X-", "", 1),
            ("YPOS", "Y", "", 2),
            # ("YNEG", "Y-", "", 3),
            ("ZPOS", "Z", "", 4),
            # ("ZNEG", "Z-", "", 5),
        ],
        name="Up Axis",
        default="YPOS",
        description="Up axis of the imported model",
    )

    hierarchy_types: bpy.props.EnumProperty(
        items=[
            ("FLAT", "Flat collection", "", 2),
            ("TREE", "Tree collection", "", 4),
            ("EMPTIES", "Parented empties", "", 6),
            # ("FLAT_AND_TREE", "Flat and tree collection", "", 0),
        ],
        name="Tree hierarchy",
        default="TREE",
        description="Organization styles of objects",
    )

    user_scale: bpy.props.FloatProperty(
        name="Scale", description="Set object scale", default=0.01, min=0.00001
    )

    # In meter. Must be multiplied by 2000 to match OCC deflection length.
    lin_deflection: bpy.props.FloatProperty(
        name="Linear Deflection",
        description="Max distance between the mesh and the theoretical shape. Smaller values increase polygon count",
        default=0.001,  # 1mm
        min=0.00001,  # 0.01mm
        unit="LENGTH",
        step=0.01,
    )

    ang_deflection: bpy.props.FloatProperty(
        name="Angular Deflection",
        description="Max angle between the tangent plane and the surrounding mesh of samples. Smaller values increase polygon count",
        default=0.0872664,  # 5°
        soft_min=0.00174532925,  # 0.1°
        min=0.000001745,  # 0.0001°
        max=math.pi,
        unit="ROTATION",
        step=100,  # 1°
        # set_transform=convert()
    )

    detail_level: bpy.props.IntProperty(
        name="Mesh Detail",
        description="How detailed you want the mesh to be",
        default=100,
        min=1,
    )

    custom_scale: bpy.props.BoolProperty(
        name="Custom Scale",
        description="Instead of loading the unit information from the file, determine it manually",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.
        # row = layout.row(align=True)

        header, body = layout.panel("Resolution", default_closed=False)
        header.label(text="General")
        if body:
            # Orientation
            row = body.row()
            row.prop(self, "up_as")

            # Hierarchy
            row = body.row()
            row.prop(self, "hierarchy_types", text="Hierarchy")

            # Custom scale
            col = body.column(align=False, heading="Overwrite Scale")
            row = col.row(align=True)
            sub = row.row(align=True)
            sub.prop(self, "custom_scale", text="")
            sub = sub.row(align=True)
            sub.active = self.custom_scale
            sub.prop(self, "user_scale", text="")

        header, body = layout.panel("Resolution", default_closed=False)
        header.label(text="Resolution")
        if body:
            # row = col.row()
            # row.prop(self, "merge_distance")

            if bpy.context.scene.stepper.simpler_parameters:
                row = body.row()
                row.prop(self, "detail_level")

            else:
                row = body.row()
                row.prop(self, "lin_deflection")

                row = body.row()
                row.prop(self, "ang_deflection")

            # row = col.row()
            # row.prop(prg, "fw_as")

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        folder = os.path.dirname(self.filepath)

        # print(type(self.files))
        # print(dir(self.files))
        l_def, a_def = self.lin_deflection * 2000, self.ang_deflection
        if bpy.context.scene.stepper.simpler_parameters:
            a_def, l_def = calculate_detail_level(self.detail_level)

        import_files = [i.name for i in self.files]

        if self.override_file != "":
            import_files = [self.override_file]

        # iterate through the selected files
        for _, i in enumerate(import_files):
            # generate full path to file
            path_to_file = os.path.join(folder, i)
            print("Opening file:", path_to_file)

            from viztracer import VizTracer
            with VizTracer(output_file="/tmp/blender_trace.json") as tracer:
                result = load_step(
                    context,
                    path_to_file,
                    custom_scale=self.user_scale if self.custom_scale else None,
                    lin_deflection=l_def,
                    ang_deflection=a_def,
                    up_as=self.up_as,
                    htypes=self.hierarchy_types,
                )
        if result:
            return {"FINISHED"}
        else:
            self.report(
                {"ERROR"}, "STEP file could not be opened. Possibly damaged file."
            )
            return {"CANCELLED"}


class STEP_OT_ClearCache(bpy.types.Operator):
    bl_idname = "object.occ_clear_cache"
    bl_label = "Clear STEP cache"
    bl_description = "Clear STEP cache, enabling the reload of a file"

    def execute(self, context):
        # utils.memorytrace_print()
        # global global_file_cache
        # items = list(global_file_cache.values())
        # for entry in items:
        #     for i, shp in enumerate(entry):
        #         label, color, tag = entry[shp]
        #         # shp.Nullify()

        global_file_cache.clear()
        return {"FINISHED"}


class STEP_OT_FixASCII(bpy.types.Operator):
    bl_idname = "object.occ_fix_ascii"
    bl_label = "Attempt STEP ASCII fix"
    bl_description = (
        "Attempt repairing invalid STEP characters.\n"
        "For files that crash the program when trying to load.\n"
        "A new file with _fix post-fix is created into the folder."
    )

    def execute(self, context):
        from pathlib import Path

        # import unicodedata

        print("Attempting to format STEP file as ASCII")
        i_file = context.scene.stepper.fix_ascii_file
        p = Path(i_file)
        if i_file == "" or not p.exists():
            self.report(
                {"ERROR"},
                "File does not exist.",
            )
            return {"FINISHED"}
        print(p.stat().st_size // 1024, "kB")

        outf = Path(p.parent, Path(p.stem.replace(" ", "_") + "_fix.step"))
        with outf.open("w", encoding="ASCII") as fo:
            with p.open("rb") as f:
                content = bytearray(f.read())
                content = content.replace(b"\r\n", b"\n")
                content = content.replace(b",\n", b",")
                content = content.replace(b"(\n", b"(")
                fo.write(content.decode("ASCII", errors="ignore"))

        self.report(
            {"INFO"},
            "Operation finished.",
        )
        return {"FINISHED"}


class STEP_OT_PrintDebug(bpy.types.Operator):
    bl_idname = "object.occ_print_debug"
    bl_label = "Print STEP debug info"
    bl_description = "Print STEP debug info"

    def execute(self, context):
        from pathlib import Path

        print("Attempting to format STEP file as ASCII")
        i_file = context.scene.stepper.print_debug
        p = Path(i_file)
        if i_file == "" or not p.exists():
            self.report(
                {"ERROR"},
                "File does not exist.",
            )
            return {"FINISHED"}

        print(p.stat().st_size // 1024, "kB")

        from . import stepanalyzer

        SA = stepanalyzer.StepAnalyzer(filename=p)
        print(SA.dump())

        self.report(
            {"INFO"},
            "Operation finished.",
        )
        return {"FINISHED"}


class STEP_OT_ReloadSTEP(bpy.types.Operator):
    bl_idname = "object.occ_reload_step"
    bl_label = "Reload STEP"
    bl_description = "Reload STEP file"

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        from . import step_reader

        filepath = context.object["STEP_file"]
        step_reader = step_reader.ReadSTEP(filepath)
        global_file_cache[filepath] = step_reader
        return {"FINISHED"}


class STEP_OT_RebuildSelected(bpy.types.Operator):
    bl_idname = "object.occ_rebuild_selected"
    bl_label = "Rebuild selected objects from the STEP file"
    bl_description = "Experimental: Causes issues on some shapes\n" + bl_label

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        prev_mode = bpy.context.mode
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        meshes = {}
        prevname = ""
        curname = ""
        build_tags = set()
        rebuilt_meshes = set()
        selected_objects = list(context.selected_objects)

        lin_def = context.scene.stepper.lin_deflection * 2000
        ang_def = context.scene.stepper.ang_deflection
        # merge_distance = context.scene.stepper.merge_distance
        if bpy.context.scene.stepper.simpler_parameters:
            ang_def, lin_def = calculate_detail_level(
                bpy.context.scene.stepper.detail_level
            )

        # select all objs with the same meshes
        for obj in selected_objects:
            for other_obj in context.scene.objects:
                if obj.data == other_obj.data:
                    other_obj.select_set(True)

        # Reload files if not in cache
        reload_needed = False
        for o in selected_objects:
            if o["STEP_file"] not in global_file_cache:
                bpy.ops.object.occ_reload_step()
                break

        # go through all selected and rebuild the meshes
        wm = bpy.context.window_manager
        wm.progress_begin(0, len(selected_objects))
        for progress_count, obj in enumerate(selected_objects):
            if obj.data.name not in meshes:
                meshes[obj.data.name] = obj.data
                sel_tag = obj["STEP_tag"]
                prevname = curname
                curname = obj["STEP_file"]
            else:
                assert meshes[obj.data.name] == obj.data

            if sel_tag in rebuilt_meshes:
                continue

            if prevname != curname:
                step_reader = global_file_cache[curname]
                # shapes_labels = step_reader.output_shapes
                tree = step_reader.tree

            for shp, node_index in tree.get_shapes():
                _, _, tag, name, _, _, _ = tree.nodes[node_index].get_values()
                if tag == sel_tag:
                    rebuilt_meshes.add(sel_tag)
                    print("Rebuilding:", sel_tag, obj.data.name)
                    obj.link(build_mesh(step_reader, name, shp, lin_def, ang_def))
                    obj.display_type = "TEXTURED"
                    build_tags.add(obj["STEP_tag"])
                    break

            wm.progress_update(progress_count)

        wm.progress_end()

        for obj in context.selected_objects:
            obj.display_type = "TEXTURED"
        
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode=prev_mode)
        return {"FINISHED"}


class STEP_PT_STEPper(bpy.types.Panel):
    bl_label = "STEPper: Build"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stepper"

    def draw(self, context):
        prg = context.scene.stepper

        layout = self.layout

        row = layout.row()
        col = row.column(align=True)

        # row = col.row()
        # row.prop(prg, "merge_distance")

        if bpy.context.scene.stepper.simpler_parameters:
            row = col.row()
            row.prop(prg, "detail_level")

        else:
            row = col.row()
            row.prop(prg, "lin_deflection")

            row = col.row()
            row.prop(prg, "ang_deflection")

        layout = self.layout
        # layout.label(text="Used memory: {}".format(total_size(global_file_cache)))
        row = layout.row()
        row.operator(STEP_OT_RebuildSelected.bl_idname, text="Rebuild selected")


class STEP_PT_STEPper_Reload(bpy.types.Panel):
    bl_label = "STEPper: File"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stepper"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(STEP_OT_ReloadSTEP.bl_idname, text="Reload STEP file")


class STEP_PT_STEPper_Debug(bpy.types.Panel):
    bl_label = "STEPper: Debug"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stepper"

    def draw(self, context):
        layout = self.layout

        bxp = layout.box()
        bxp.label(text="Enforce ASCII")
        col = bxp.row().column(align=True)

        prg = context.scene.stepper
        row = col.row()
        row.prop(prg, "fix_ascii_file")
        row = col.row()
        row.operator("object.occ_fix_ascii", text="Attempt fix STEP charset")

        # row = layout.row()
        # row.label(text="Error messages:")

        if (
            context.object is not None
            and "STEP_file" in context.object
            and context.object["STEP_file"] in global_file_cache
        ):
            bxp = layout.box()
            bxp.label(text="Reported problems:")

            row = bxp.row()
            col = row.column(align=True)
            step_reader = global_file_cache[context.object["STEP_file"]]
            for k, v in step_reader.import_problems.items():
                row = col.row()
                row.label(text=k + ": " + repr(v))

            bxs = layout.box()
            bxs.label(text="Skipped shapes:")

            row = bxs.row()
            col = row.column(align=True)
            if len(step_reader.skipped_shapes) > 0:
                for v in step_reader.skipped_shapes:
                    row = col.row()
                    row.label(text=repr(v))
            else:
                row = col.row()
                row.label(text="No skipped shapes")

        else:
            bxp = layout.box()
            row = bxp.row()
            row.label(text="Select active STEP object")


class STEP_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(bpy.context.scene.stepper, "build_materials")

        row = layout.row()
        row.prop(bpy.context.scene.stepper, "hack_skip_zero_solids")

        row = layout.row()
        row.prop(bpy.context.scene.stepper, "simpler_parameters")

        # row = layout.row()
        # row.prop(bpy.context.scene.stepper, "hierarchy_types")


def menu_func_import(self, context):
    self.layout.operator(
        STEP_OT_ImportStepCADOperator.bl_idname, text="STEP (.step, .stp)"
    )


classes = (
    PG_Stepper,
    STEP_OT_ImportStepCADOperator,
    STEP_OT_ClearCache,
    STEP_OT_RebuildSelected,
    STEP_OT_ReloadSTEP,
    STEP_OT_FixASCII,
    STEP_OT_PrintDebug,
    STEP_PT_STEPper,
    STEP_PT_STEPper_Reload,
    STEP_PT_STEPper_Debug,
    STEP_AddonPreferences,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.stepper = bpy.props.PointerProperty(type=PG_Stepper)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for c in classes[::-1]:
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    del bpy.types.Scene.stepper


if __package__ == "__main__":
    register()
