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
# Modified 2025 Romain Guimbal

import os
import math
import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ImportHelper
from .object_generator import (
    HierarchyType,
    load_step,
    build_mesh,
    GLOBAL_FILE_CACHE,
)
from .utils import calculate_deflections_for_artist_friendly

axis_enum = [
    ("XPOS", "X", "", 0),
    # ("XNEG", "X-", "", 1),
    ("YPOS", "Y", "", 2),
    # ("YNEG", "Y-", "", 3),
    ("ZPOS", "Z", "", 4),
    # ("ZNEG", "Z-", "", 5),
]

# In blender unit.
lin_deflection_prop = bpy.props.FloatProperty(
    name="Linear Deflection",
    description="Max distance between the mesh and the theoretical shape. Smaller values increase polygon count. In Blender unit",
    default=0.002,  # 2mm (if STEP:mm  and blender:m)
    soft_min=0.00001,  # 0.01mm
    unit="LENGTH",
    step=0.01,
)

ang_deflection_prop = bpy.props.FloatProperty(
    name="Angular Deflection",
    description="Max angle between the tangent plane and the surrounding mesh of samples. Smaller values increase polygon count",
    default=0.0872664,  # 5° (in radian)
    soft_min=0.00174532925,  # 0.1°
    min=0.000001745,  # 0.0001°
    max=math.pi,
    unit="ROTATION",
    step=100,  # 1° (scalar)
)

detail_level_prop = bpy.props.IntProperty(
    name="Mesh Detail",
    description="Higher values means higher polygon count",
    default=100,
    min=1,
)


class STEP_PG_properties(bpy.types.PropertyGroup):
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
        description="More intuitive parameters for visual look",
        default=False,
    )

    detail_level: detail_level_prop

    lin_deflection_bl_unit: lin_deflection_prop

    ang_deflection: ang_deflection_prop

    fix_ascii_file: bpy.props.StringProperty(
        name="File",
        description="Path to problematic STEP file",
        default="",
        maxlen=1024,
        subtype="FILE_PATH",
    )

    preferred_up_axis: bpy.props.EnumProperty(
        items=axis_enum,
        name="preferred Up Axis",
        default=axis_enum[2][0],
        description="Preselected up axis at import",
    )


class STEP_OT_ImportStepCADOperator(bpy.types.Operator, ImportHelper):
    bl_idname = "wm.occ_import_step"
    bl_label = "Import STEP"
    bl_description = "Import a STEP file"
    bl_options = {"PRESET"}

    filter_glob: StringProperty(default="*.step;*.stp;*.st", options={"HIDDEN"})
    files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)
    override_file: StringProperty(default="", options={"HIDDEN"})

    fw_as: bpy.props.EnumProperty(
        items=axis_enum,
        name="Forward",
        default="YPOS",
        description="Forward axis of the imported model",
    )

    up_as: bpy.props.EnumProperty(
        items=axis_enum,
        name="Up Axis",
        description="Up axis of the imported model",
        default="ZPOS",
    )

    hierarchy_type: bpy.props.EnumProperty(
        items=[
            (str(int(HierarchyType.COLLECTION_FLAT)), "Flat collection", "", 0),
            (str(int(HierarchyType.COLLECTION_TREE)), "Tree collection", "", 1),
            (str(int(HierarchyType.EMPTIES_TREE)), "Parented empties", "", 2),
            # (str(int(HierarchyType.COLLECTION_FLAT_AND_TREE)), "Flat and tree collection", "", 3),
            (
                str(int(HierarchyType.COLLECTION_INSTANCES)),
                "Collection instances",
                "",
                4,
            ),
        ],
        name="Tree hierarchy",
        default=str(int(HierarchyType.COLLECTION_TREE)),
        description="Organization style of objects",
    )

    scale_overwritten: bpy.props.FloatProperty(
        name="Scale", description="Set object scale", default=0.001, soft_min=0.00001
    )

    lin_deflection_bl_unit: lin_deflection_prop

    ang_deflection: ang_deflection_prop

    detail_level: detail_level_prop

    custom_scale: bpy.props.BoolProperty(
        name="Custom Scale",
        description="Instead of loading the unit information from the file, determine it manually",
        default=False,
    )

    ran: bpy.props.BoolProperty(
        name="Operator Already ran in this session",
    )

    def invoke(self, context, event):
        if not self.ran:
            self.up_as = context.scene.stepper.preferred_up_axis
            self.ran = True
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.
        # row = layout.row(align=True)

        header, body = layout.panel("General", default_closed=False)
        header.label(text="General")
        if body:
            # Orientation
            row = body.row()
            row.prop(self, "up_as", expand=True)

            # Hierarchy
            row = body.row()
            row.prop(self, "hierarchy_type", text="Hierarchy")

            # Custom scale
            col = body.column(align=False, heading="Overwrite Scale")
            row = col.row(align=True)
            sub = row.row(align=True)
            sub.prop(self, "custom_scale", text="")
            sub = sub.row(align=True)
            sub.active = self.custom_scale
            sub.prop(self, "scale_overwritten", text="")

        header, body = layout.panel("Resolution", default_closed=False)
        header.label(text="Resolution")
        if body:
            # row = col.row()
            # row.prop(self, "merge_distance")
            col = body.column()
            if bpy.context.scene.stepper.simpler_parameters:
                col.prop(self, "detail_level")
            else:
                col.prop(self, "lin_deflection_bl_unit")
                col.prop(self, "ang_deflection")

            # row = col.row()
            # row.prop(prg, "fw_as")

    def execute(self, context):
        result = None
        l_def = self.lin_deflection_bl_unit
        a_def = self.ang_deflection
        if bpy.context.scene.stepper.simpler_parameters:
            a_def, l_def = calculate_deflections_for_artist_friendly(self.detail_level)

        import_files = [i.name for i in self.files]

        if self.override_file != "":
            import_files = [self.override_file]

        # --- VizTracer Block ---

        # from viztracer import VizTracer

        # with VizTracer(
        #     output_file="/tmp/blender_trace.json",
        #     tracer_entries=3000000,
        #     min_duration=5,
        # ) as tracer:

        folder = os.path.dirname(self.filepath)

        # iterate through the selected files
        for _, i in enumerate(import_files):
            # generate full path to file
            filepath = os.path.join(folder, i)
            print("Opening file:", filepath)
            result = load_step(
                context,
                filepath,
                custom_scale=self.scale_overwritten if self.custom_scale else None,
                lin_deflection=l_def,
                ang_deflection=a_def,
                up_as=self.up_as,
                htypes=HierarchyType(int(self.hierarchy_type)),
            )

        # --- End of VizTracer block ---

        if result is not None:
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
        GLOBAL_FILE_CACHE.clear()
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
        import unicodedata

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

class STEP_OT_ReloadSTEP(bpy.types.Operator):
    bl_idname = "object.occ_reload_step"
    bl_label = "Reload STEP"
    bl_description = "Reload STEP file"

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        from . import importer

        filepath = context.object["STEP_file"]
        step_reader = importer.ReadSTEP(filepath)
        GLOBAL_FILE_CACHE[filepath] = step_reader
        return {"FINISHED"}


class STEP_OT_RebuildSelected(bpy.types.Operator):
    bl_idname = "object.occ_rebuild_selected"
    bl_label = "Rebuild selected objects from the STEP file"
    bl_description = "Experimental: Causes issues on some shapes\n" + bl_label

    @classmethod
    def poll(cls, context):
        return context.object is not None and "STEP_file" in context.object

    def execute(self, context):
        meshes = {}
        prevname = ""
        curname = ""
        build_tags = set()
        rebuilt_meshes = set()
        selected_objects = list(context.selected_objects)

        prev_mode = context.mode
        bpy.ops.object.mode_set(mode="OBJECT")

        lin_def = context.scene.stepper.lin_deflection_bl_unit
        ang_def = context.scene.stepper.ang_deflection

        # merge_distance = context.scene.stepper.merge_distance
        if bpy.context.scene.stepper.simpler_parameters:
            ang_def, lin_def = calculate_deflections_for_artist_friendly(
                bpy.context.scene.stepper.detail_level
            )

        # select all objs with the same meshes
        for obj in selected_objects:
            for other_obj in context.scene.objects:
                if obj.data == other_obj.data:
                    other_obj.select_set(True)

        # Reload files if not in cache
        for o in selected_objects:
            if o["STEP_file"] not in GLOBAL_FILE_CACHE:
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
                step_reader = GLOBAL_FILE_CACHE[curname]
                # shapes_labels = step_reader.output_shapes
                tree = step_reader.tree

            for shp, node_index in tree.get_shapes():
                _, _, tag, name, _, _, _ = tree.nodes[node_index].get_values()
                if tag == sel_tag:
                    rebuilt_meshes.add(sel_tag)
                    print("Rebuilding:", sel_tag, obj.data.name)
                    build_mesh(step_reader, obj, shp, lin_def, ang_def)
                    obj.display_type = "TEXTURED"
                    build_tags.add(obj["STEP_tag"])
                    break

            wm.progress_update(progress_count)

        wm.progress_end()

        for obj in context.selected_objects:
            obj.display_type = "TEXTURED"

        bpy.ops.object.mode_set(mode=prev_mode)

        return {"FINISHED"}


class STEP_PT_side_panel(bpy.types.Panel):
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
            row.prop(prg, "lin_deflection_bl_unit")

            row = col.row()
            row.prop(prg, "ang_deflection")

        layout = self.layout
        # layout.label(text="Used memory: {}".format(total_size(GLOBAL_FILE_CACHE)))
        row = layout.row()
        row.operator(STEP_OT_RebuildSelected.bl_idname, text="Rebuild selected")


class STEP_PT_side_panel_Reload(bpy.types.Panel):
    bl_label = "STEPper: File"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Stepper"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.operator(STEP_OT_ReloadSTEP.bl_idname, text="Reload STEP file")


class STEP_PT_side_panel_Debug(bpy.types.Panel):
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
            and context.object["STEP_file"] in GLOBAL_FILE_CACHE
        ):
            bxp = layout.box()
            bxp.label(text="Reported problems:")

            row = bxp.row()
            col = row.column(align=True)
            step_reader = GLOBAL_FILE_CACHE[context.object["STEP_file"]]
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

        col = layout.column()
        col.prop(bpy.context.scene.stepper, "build_materials")
        col.prop(bpy.context.scene.stepper, "hack_skip_zero_solids")
        col.prop(bpy.context.scene.stepper, "simpler_parameters")

        row = col.row(align=True)  # does nothing. Probably unsuported
        row.label(text="Preferred Up Axis")
        row.prop(bpy.context.scene.stepper, "preferred_up_axis", expand=True)

        # col = layout.col()
        # col.prop(bpy.context.scene.stepper, "hierarchy_type")

        # col.operator(PMM_OT_EnsurePIP.bl_idname, text="Ensure PIP")
        # col.operator(PMM_OT_UpgradePIP.bl_idname, text="Upgrade PIP")
        # col.operator(PMM_OT_PIPList.bl_idname, text="List")


def menu_func_import(self, context):
    self.layout.operator(
        STEP_OT_ImportStepCADOperator.bl_idname, text="STEP (.step, .stp)"
    )


classes = (
    STEP_PG_properties,
    STEP_OT_ImportStepCADOperator,
    STEP_OT_ClearCache,
    STEP_OT_RebuildSelected,
    STEP_OT_ReloadSTEP,
    STEP_OT_FixASCII,
    STEP_PT_side_panel,
    STEP_PT_side_panel_Reload,
    STEP_PT_side_panel_Debug,
    STEP_AddonPreferences,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.stepper = bpy.props.PointerProperty(type=STEP_PG_properties)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    for c in classes[::-1]:
        bpy.utils.unregister_class(c)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    del bpy.types.Scene.stepper
