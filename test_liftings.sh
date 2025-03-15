project_name="LIFTING_TEST"
DATASETS=('MUTAG') #'PROTEINS'



# # ------------------------- Graph2Combinatorial -------------------------
transforms_experiments=(exp_combinatorial/g2c_simplicial_paths)
# Date 25/02 
# working: 
# Not working: exp_combinatorial/g2c_curve, exp_combinatorial/g2c_ring_close_atoms, exp_combinatorial/g2c_simplicial_paths
# NOTE 1: I'm pretty sure exp_combinatorial/g2c_curve is working but it is way too slow, even on MUTAG
# NOTE 2: exp_combinatorial/g2c_ring_close_atoms works only on data with data.smiles attribute, which is not the case for our datasets
# NOTE 3: exp_combinatorial/g2c_simplicial_paths is not working. They don't check if they find simplices of high enough ranks (they need cliques of adeguate size to be able to find them) so everything breaks.
# NOTE 4: last commit before deletion 2cc98ffaa64508dc22ba8dc381112c573c1dfa5c
# NOTE 5: commit when deleted 86e53397a5cec8aa891acfd1f2ea61f9af247502



# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=cell/topotune\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done


# # ------------------------- Hypergraph2Simplicial -------------------------
# transforms_experiments=(exp_simplicial/h2s_heat)
# # Date 25/02 
# # working: 
# # Not working exp_simplicial/h2s_heat
# # NOTE 1: exp_simplicial/h2s_heat is not working. We can remove the dependency from the external library but the lifting doesn't return a simplicial complex but a torch_geometric.data.Data object.
# # NOTE 2: last commit before deletion 2cc98ffaa64508dc22ba8dc381112c573c1dfa5c
# # NOTE 3: commit when deleted 86e53397a5cec8aa891acfd1f2ea61f9af247502

# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=simplicial/scn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done


# # ------------------------- Graph2Hypergraph -------------------------
# transforms_experiments=(exp_hypergraph/g2h_forman_ricci_curvature exp_hypergraph/g2h_expander_graph g2h_kernel  exp_hypergraph/g2h_khop g2h_knn exp_hypergraph/g2h_mapper exp_hypergraph/g2h_modularity_maximization)
# # Date 25/02 
# # working: exp_hypergraph/g2h_forman_ricci_curvature exp_hypergraph/g2h_expander_graph g2h_kernel  exp_hypergraph/g2h_khop g2h_knn exp_hypergraph/g2h_mapper exp_hypergraph/g2h_modularity_maximization
# # Not working 
# # NOTE 1: 


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=hypergraph/unignn2\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done

# # ------------------------- Graph2CELLULAR -------------------------
# transforms_experiments=(exp_cell/g2c_discrete_configuration_complex)

# # Date 24/02
# # working: exp_cell/g2h_discrete_configuration_complex
# # Not working:
# # NOTE 1: The discrete_configuration_complex has a


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=cell/cwn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#     done
# done

# ------------------------- Graph2SIMPLICIAL -------------------------
# transforms_experiments=(exp_simplicial/g2h_neighborhood_complex)
# # Date 24/02
# # working: g2s_khop g2s_vietoris_rips g2h_neighborhood_complex g2s_graph_induceds g2s_dnd

# # NOTE 1: the code of this submission is good but g2s_line (line lifting) construct line graph and finds graph induced topology. Hence the edges becomes nodes. THis lifting do not fits into our TB framework without modification 
# # NOTE 2: it seems that g2s_graph_induced is our clique lifting. 
# # NOTE 3: g2s_dnd is very slow as it generates enormous amout of simplices, even for small datasets.

# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=simplicial/scn\
#             dataset=graph/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#     done
# done

# # ------------------------- Pointcloud2Hypergraph -------------------------
# transforms_experiments=(exp_hypergraph/p2h_mogmst exp_hypergraph/p2h_pointnet exp_hypergraph/p2h_voronoi)
# DATASETS=('geometric_shapes')
# # Date 25/02 
# # working:exp_hypergraph/p2h_mogmst exp_hypergraph/p2h_pointnet exp_hypergraph/p2h_voronoi
# # Not working: 
# # NOTE 1: 


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=hypergraph/unignn2\
#             dataset=pointcloud/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done

# # ------------------------- Pointcloud2Simplicial -------------------------
# transforms_experiments=(exp_simplicial/p2s_alpha_complex exp_simplicial/p2s_alpha_complex_delaunay exp_simplicial/p2s_random_flag_complex)
# DATASETS=('geometric_shapes')
# # Date 25/02 
# # working: exp_simplicial/p2s_alpha_complex exp_simplicial/p2s_alpha_complex_delaunay exp_simplicial/p2s_random_flag_complex
# # Not working:
# # NOTE 1: exp_simplicial/p2s_delaunay is equal to exp_simplicial/p2s_alpha_complex_delaunay so I removed it (commit f7025bb7fed49821e1cc47f4c7c9ebf97e2040dd)


# for transform in ${transforms_experiments[*]}
# do
#     for dataset in ${DATASETS[*]}
#     do
#             python topobench/run.py\
#             model=simplicial/scn\
#             dataset=pointcloud/$dataset\
#             trainer.max_epochs=2\
#             trainer.min_epochs=1\
#             trainer.check_val_every_n_epoch=1\
#             transforms=$transform\
#             logger.wandb.project=$project_name
#             #trainer.devices=\[$device\]
#     done
# done

# ------------------------- Hypergraph2Combinatorial -------------------------
transforms_experiments=(exp_combinatorial/h2c_universal_strict)
# Date 25/02 
# working: exp_combinatorial/h2c_universal_strict
# Not working: 
# NOTE 1: 



for transform in ${transforms_experiments[*]}
do
    for dataset in ${DATASETS[*]}
    do
            python topobench/run.py\
            model=cell/topotune\
            dataset=hypergraph/cocitation_cora\
            trainer.max_epochs=2\
            trainer.min_epochs=1\
            trainer.check_val_every_n_epoch=1\
            transforms=$transform\
            logger.wandb.project=$project_name
            #trainer.devices=\[$device\]
    done
done
