




# ---WORK---
# python -m topobenchmark model=simplicial/scn dataset=graph/PROTEINS trainer.max_epochs=2 trainer.min_epochs=1 trainer.check_val_every_n_epoch=1 transforms=g2s_khop

# ISSUE: This one has raised an error with omegaconf and resolver. Basically some resolver asks for "complex_dim" in lifting config. I put complex_dim=1 and it worked, however in general there shouldn't be such issue.
# python -m topobenchmark model=hypergraph/unignn dataset=graph/PROTEINS trainer.max_epochs=2 trainer.min_epochs=1 trainer.check_val_every_n_epoch=1 transforms=g2h_expander


# ISSUE: very slow, was not able to finish even on PRORTEINS, hence maybe work maybe not.
# python -m topobenchmark model=simplicial/scn dataset=graph/PROTEINS trainer.max_epochs=2 trainer.min_epochs=1 trainer.check_val_every_n_epoch=1 transforms=g2s_graph_induced_lifting # VERY SLOW

# ---DO NOT WORK---
# python -m topobenchmark model=simplicial/scn dataset=graph/PROTEINS trainer.max_epochs=2 trainer.min_epochs=1 trainer.check_val_every_n_epoch=1 transforms=g2s_line_lifting
# python -m topobenchmark model=simplicial/scn dataset=graph/PROTEINS trainer.max_epochs=2 trainer.min_epochs=1 trainer.check_val_every_n_epoch=1 transforms=g2s_vietoris_rips_lifting

