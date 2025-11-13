"""Dataset class for US County Demographics dataset."""

import os
import os.path as osp
import shutil
from typing import ClassVar

import numpy as np
import pandas as pd
import polars as pl
import toponetx as tnx
from hnne.finch_clustering import FINCH
from omegaconf import DictConfig
import torch
from torch_geometric.data import Data, InMemoryDataset, extract_zip
from torch_geometric.io import fs

from topobench.data.utils import get_colored_hypergraph_connectivity


class TwitterArlequinDataset(InMemoryDataset):
    r"""Dataset class for Twitter Arlequin dataset.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be saved.
    name : str
        Name of the dataset.
    parameters : DictConfig
        Configuration parameters for the dataset.

    Attributes
    ----------
    URLS (dict): Dictionary containing the URLs for downloading the dataset.
    FILE_FORMAT (dict): Dictionary containing the file formats for the dataset.
    RAW_FILE_NAMES (dict): Dictionary containing the raw file names for the dataset.
    """

    URLS: ClassVar = {}

    FILE_FORMAT: ClassVar = {}

    RAW_FILE_NAMES: ClassVar = {
        "raw_data": "/data/gbg141/Arlequin/data/twitter_dataset.csv",
        "embedded_data": "/data/gbg141/Arlequin/data/twitter_embedded.parquet",
        "processed_data": "/data/gbg141/Arlequin/data/twitter_s10k.parquet"
    }

    def __init__(
        self,
        root: str,
        name: str,
        parameters: DictConfig,
    ) -> None:
        self.name = name
        self.parameters = parameters
        super().__init__(
            root,
        )

        out = fs.torch_load(self.processed_paths[0])
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

        assert isinstance(self._data, Data)

    def __repr__(self) -> str:
        return f"{self.name}(self.root={self.root}, self.name={self.name}, self.parameters={self.parameters}, self.force_reload={self.force_reload})"

    @property
    def raw_dir(self) -> str:
        """Return the path to the raw directory of the dataset.

        Returns
        -------
        str
            Path to the raw directory.
        """
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory of the dataset.

        Returns
        -------
        str
            Path to the processed directory.
        """

        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the raw file names for the dataset.

        Returns
        -------
        list[str]
            List of raw file names.
        """
        return []

    @property
    def processed_file_names(self) -> str:
        """Return the processed file name for the dataset.

        Returns
        -------
        str
            Processed file name.
        """
        return "data.pt"
    
    def create_posts_nodes(self, df):
        """Create post nodes from dataframe and add to complex.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing the twitter data.
        """
        self.posts = []
        for idx in range(len(df)):
            # get values
            id = df.at[idx,"id"]
            msg = df.at[idx,"message"]
            user_id = df.at[idx, "user_id"]
            replies = df.at[idx, "reply_to_user_id"].split("'")[1::2]
            replies = [int(r) for r in replies]
            mentions = df.at[idx, "mentions_user_id"].split("'")[1::2]
            mentions = [int(m) for m in mentions]

            # create post object and add to chg
            post = Post(id, msg, user_id, replies, mentions)
            self.posts.append(post)
            self.complex.add_node(post)

    def build_user_hyperedges(self, rank=1):
        """Build user hyperedges from posts and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 1.
        """
        # find posts by each user
        self.users_to_posts = dict()
        for user in self.user_ids:
            self.users_to_posts[user] = []
        for post in self.posts:
            self.users_to_posts[post.user_id].append(post)

        # create users as hyperedges
        ids_to_users = dict()
        for user in self.user_ids:
            user = tnx.HyperEdge(self.users_to_posts[user], rank=rank)
            ids_to_users[user] = user
            self.complex.add_cell(user, rank=rank)

    def build_connection_hyperedges(self, rank=2):
        """Build user connection hyperedges from posts and add to complex.
        
        Parameters
        ----------
        rank : int, optional
            Rank of the hyperedges, by default 2.
        """
        self.replies = []
        self.mentions = []
        for post in self.posts:
            for r in post.reply_to_user_id:
                if r in self.user_ids and post.user_id != r:
                    combined_posts = self.users_to_posts[post.user_id] + self.users_to_posts[r]
                    reply = tnx.HyperEdge(combined_posts, rank=rank)
                    self.replies.append(combined_posts)
                    self.complex.add_cell(reply, rank=rank)
            for m in post.mentions_user_id:
                if m in self.user_ids and post.user_id != m:
                    combined_posts = self.users_to_posts[post.user_id] + self.users_to_posts[m]
                    mention = tnx.HyperEdge(combined_posts, rank=rank)
                    self.mentions.append(combined_posts)
                    self.complex.add_cell(mention, rank=rank)

    def cluster_posts(self, embeddings, cluster_level_posts=2, random_seed=57):
        """Cluster posts based on embeddings and add semantic hyperedges to complex.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array containing the post embeddings.
        cluster_level_posts : int, optional
            Clustering level for posts, by default 2.
        random_seed : int, optional
            Random seed for clustering, by default 57.
        """
        clusters, n_clusters, _, _ = FINCH(data=embeddings, distance="cosine", verbose=0, random_state=random_seed)
        print("Embeddings: ", n_clusters)
        self.semantics = clusters[:, -cluster_level_posts]
        self.n_clusters_posts = n_clusters[-cluster_level_posts]

    def build_semantic_hyperedges(self):
        """Build semantic hyperedges from clustered posts and add to complex."""
        self.semantic_hyperedges = []
        for label in range(self.n_clusters_posts):
            # get users in cluster
            mask = self.semantics == label
            cluster_posts = np.array(self.posts)[mask]
            # create hyperedge
            cluster = tnx.HyperEdge(cluster_posts, rank=4)
            self.semantic_hyperedges.append(cluster_posts)
            self.complex.add_cell(cluster, rank=4)

    def compute_and_cluster_user_feature_vectors(self, cluster_level_users=0, random_seed=57):
        """Compute feature vectors for each user based on clustered posts.
        
        Parameters
        ----------
        embedding_ids : list
            List of embedding IDs for the posts.
        cluster_level_users : int, optional
            Clustering level for users, by default 0.
        random_seed : int, optional
            Random seed for clustering, by default 57.
        """
        feature_vectors = []
        for user in self.user_ids:
            feature = [0.0 for _ in range(self.n_clusters_posts)]
            for post in self.users_to_posts[user]:
                index = self.posts_ids.index(post.id)
                cluster = self.semantics[index]
                feature[cluster] += 1
            feature_vectors.append([f / sum(feature) for f in feature])
        self.feature_vectors = np.array(feature_vectors)
        
        # cluster feature vectors
        clusters, n_clusters, _, _ = FINCH(data=self.feature_vectors, distance="euclidean", verbose=0, random_state=random_seed)
        print("Users: ", n_clusters)
        self.user_clusters = clusters[:, cluster_level_users]
        self.n_clusters_users = n_clusters[cluster_level_users]
        
    def build_community_hyperedges(self):
        """Build community hyperedges from clustered users and add to complex."""
        self.community_hyperedges = []
        for label in range(self.n_clusters_users):
            # get users in cluster
            mask = self.user_clusters == label
            cluster_users = np.array(self.user_ids)[mask]
            # get posts of users
            cluster_posts = []
            for user in cluster_users:
                cluster_posts.extend(self.users_to_posts[user])
            # create hyperedge
            cluster = tnx.HyperEdge(cluster_posts, rank=3)
            self.community_hyperedges.append(cluster_posts)
            self.complex.add_cell(cluster, rank=3)
            
    def get_connectivity(self):
        """Get connectivity of the complex."""
        # parameters
        cluster_lvl_posts = 1 # cluster level for posts
        cluster_lvl_users = 0 # cluster level for users
        max_rank = 4 # maximum rank of the complex
        random_seed = 57

        # open dataset
        df = pd.read_parquet(self.RAW_FILE_NAMES["processed_data"])
        df_embeddings = pl.read_parquet(self.RAW_FILE_NAMES["embedded_data"]).to_pandas()
        
        # store user and post ids
        self.user_ids = list(set(df["user_id"]))
        self.posts_ids = df["id"].to_list()

        # set dfs to same orientation
        df_embeddings.set_index("id", inplace=True)
        df_embeddings = df_embeddings.loc[self.posts_ids]
        embeddings = np.array(df_embeddings["embedding"].to_list())

        # create empty colored hypergraph complex
        self.complex = tnx.ColoredHyperGraph()

        # create posts as nodes
        self.create_posts_nodes(df)
        
        # build user hyperedges
        self.build_user_hyperedges(rank=1)

        # create user connections as hyperedges
        self.build_connection_hyperedges(rank=2)

        # cluster posts based on embeddings and create hyperedge for semantic clusters
        self.cluster_posts(embeddings, cluster_level_posts=cluster_lvl_posts, random_seed=random_seed)
        self.build_semantic_hyperedges()

        # compute feature vectors for each user
        self.compute_and_cluster_user_feature_vectors(cluster_level_users=cluster_lvl_users, random_seed=random_seed)
        self.build_community_hyperedges()

        connectivity = get_colored_hypergraph_connectivity(self.complex, max_rank=max_rank)
        return connectivity, embeddings
    
    def get_node_features_and_labels(self, embeddings):
        """Get node features and labels for the dataset.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Array containing the post embeddings.

        Returns
        -------
        features_and_labels : dict
            Dictionary containing the features and labels for each node.
        """
        features_and_labels = dict()
        features_and_labels["x_0"] = torch.tensor(embeddings, dtype=torch.float32)
        features_and_labels["x_1"] = torch.tensor(self.feature_vectors, dtype=torch.float32)
        # features_and_labels["x_2"] = torch.zeros_like(features_and_labels["x_0"])
        features_and_labels["x_3"] = torch.zeros_like(features_and_labels["x_0"])
        features_and_labels["y"] = torch.tensor(self.semantics, dtype=torch.long)
        return features_and_labels

    def process(self) -> None:
        r"""Handle the data for the dataset.
        """
        connectivity, embeddings = self.get_connectivity()
        features_and_labels = self.get_node_features_and_labels(embeddings)
        data = Data(**connectivity, **features_and_labels)

        data_list = [data]
        self.data, self.slices = self.collate(data_list)
        self._data_list = None  # Reset cache.
        fs.torch_save(
            (self._data.to_dict(), self.slices, {}, self._data.__class__),
            self.processed_paths[0],
        )


# node class
class Post:
    def __init__(self, id, msg, user_id, reply_to_user_id, mentions_user_id):
        self.id = id
        self.msg = msg
        self.user_id = user_id
        self.reply_to_user_id = reply_to_user_id
        self.mentions_user_id = mentions_user_id
        
    def __repr__(self):
        return f"Post(id={self.id}, user_id={self.user_id})"

    def __lt__(self, other):
        return self.msg < other.msg

