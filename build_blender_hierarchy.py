import bpy
from .utils import set_obj_matrix_world


def build_flat_collection(
    filename,
    tree,
    created_objs,
):
    flat_collection = bpy.data.collections.new(filename + ".flat")
    bpy.context.scene.collection.children.link(flat_collection)

    created_collections = {}
    for obj in created_objs:
        group_name = obj["STEP_name"]

        # max collection name len = 61
        if len(group_name) > 50:
            group_name = group_name[:25] + "_" + group_name[-25:]

        # TODO: check dupe collections for dupe imports
        if group_name not in created_collections:
            group_collection = bpy.data.collections.new(group_name)
            created_collections[group_name] = group_collection
            flat_collection.children.link(group_collection)
        else:
            group_collection = created_collections[group_name]

        global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
        set_obj_matrix_world(obj, global_t)
        group_collection.objects.link(obj)


def build_collection_tree(filename, tree, created_objs):
    tree_collection = bpy.data.collections.new(filename + ".hierarchy")
    bpy.context.scene.collection.children.link(tree_collection)
    hierarchy_collections = {}
    hierarchy_collections[-1] = tree_collection

    def node_parse(node, level, parent_collection):
        # if "name" in node and node["children"] is not None:
        if len(node.children) > 0:
            collection_node = bpy.data.collections.new(node.name)
            assert node.index not in hierarchy_collections
            hierarchy_collections[node.index] = collection_node

            parent_collection.children.link(collection_node)
            for c in node.children:
                node_parse(tree.nodes[c], level + 1, collection_node)

    root = tree.nodes[0]
    if len(root.children) > 0:
        for c in root.children:
            node_parse(tree.nodes[c], 0, tree_collection)

        # link objects to tree
        if len(hierarchy_collections.items()) > 0:
            for obj in created_objs:
                hierarchy_collections[obj["STEP_parent"]].objects.link(obj)
                global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
                set_obj_matrix_world(obj, global_t)


def build_empty_tree(tree, created_objs, created_uuid):
    for obj in created_objs:
        global_t = tree.nodes[obj["STEP_tree_location"]].global_transform
        set_obj_matrix_world(obj, global_t)
        bpy.context.scene.collection.objects.link(obj)

        # Parent objs
        parent_id = obj["STEP_parent"]
        if parent_id in created_uuid:
            parent = created_uuid[parent_id]
            obj.parent = parent
            obj.matrix_parent_inverse = parent.matrix_world.inverted()


def build_blender_hierarchy(
    filename,
    tree,
    created_objs,
    hierarchy_flat,
    hierarchy_tree,
    hierarchy_empties,
    created_uuid,
):
    # build flat collection
    if hierarchy_flat:
        build_flat_collection(filename, tree, created_objs)

    # build tree of collections
    if hierarchy_tree:
        build_collection_tree(filename, tree, created_objs)

    # build hierarchy with empties
    if hierarchy_empties:
        build_empty_tree(tree, created_objs, created_uuid)
