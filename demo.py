"""Demo script to explore the constructed graph"""

import numpy as np
import pandas as pd
import torch

import config


def load_graph():
    """Load the constructed heterogeneous graph"""
    print("Loading graph...")
    graph = torch.load(f"{config.GRAPH_DATA_PATH}hetero_graph.pt", weights_only=False)
    mappings = torch.load(
        f"{config.GRAPH_DATA_PATH}entity_mappings.pt", weights_only=False
    )
    return graph, mappings


def explore_job(graph, mappings, job_idx: int):
    """Explore a specific job and its connections"""
    job_id = graph["job"].job_ids[job_idx]

    print("\n" + "=" * 70)
    print(f"EXPLORING JOB: {job_id}")
    print("=" * 70)

    # Load original data for details
    df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")
    job_data = df[df["JobID"] == job_id].iloc[0]

    print("\n📋 BASIC INFO:")
    print(f"  Title: {job_data['Title']}")
    print(f"  Company: {job_data['Name company']}")
    print(f"  Location: {job_data['location_clean']}")
    print(f"  Salary: {job_data['Salary']}")
    print(f"  Experience: {job_data['Experience']}")

    # Find company connection
    edge_index = graph[("job", "posted_by", "company")].edge_index
    company_idx = edge_index[1, edge_index[0] == job_idx][0].item()
    company_name = graph["company"].company_names[company_idx]

    print("\n🏢 COMPANY CONNECTION:")
    print(f"  Connected to: {company_name} (idx: {company_idx})")

    # Count jobs from same company
    same_company_jobs = (edge_index[1] == company_idx).sum().item()
    print(f"  Total jobs from this company: {same_company_jobs}")

    # Find similar jobs
    edge_index = graph[("job", "similar_to", "job")].edge_index
    similar_jobs_mask = edge_index[0] == job_idx
    similar_job_indices = edge_index[1, similar_jobs_mask].tolist()

    print(f"\n🔗 SIMILAR JOBS: (Found {len(similar_job_indices)} similar jobs)")

    # Load similarity matrix for scores
    similarity_matrix = np.load(f"{config.PROCESSED_DATA_PATH}similarity_matrix.npy")

    # Show top 5 most similar
    similarities = [
        (idx, similarity_matrix[job_idx, idx]) for idx in similar_job_indices
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    for i, (sim_idx, sim_score) in enumerate(similarities[:5], 1):
        sim_job_id = graph["job"].job_ids[sim_idx]
        sim_job_data = df[df["JobID"] == sim_job_id].iloc[0]
        print(f"\n  {i}. Similarity: {sim_score:.3f}")
        print(f"     Job: {sim_job_data['Title'][:60]}")
        print(f"     Company: {sim_job_data['Name company'][:40]}")


def explore_company(graph, mappings, company_idx: int):
    """Explore a specific company and its job postings"""
    company_name = graph["company"].company_names[company_idx]

    print("\n" + "=" * 70)
    print(f"EXPLORING COMPANY: {company_name}")
    print("=" * 70)

    # Find all jobs from this company
    edge_index = graph[("company", "posts", "job")].edge_index
    job_indices = edge_index[1, edge_index[0] == company_idx].tolist()

    print("\n📊 COMPANY STATISTICS:")
    print(f"  Total jobs posted: {len(job_indices)}")

    # Load job data
    df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")

    print("\n💼 JOB LISTINGS:")
    for i, job_idx in enumerate(job_indices[:10], 1):  # Show first 10
        job_id = graph["job"].job_ids[job_idx]
        job_data = df[df["JobID"] == job_id].iloc[0]
        print(f"\n  {i}. {job_data['Title'][:60]}")
        print(f"     Salary: {job_data['Salary']}")
        print(f"     Location: {job_data['location_clean']}")

    if len(job_indices) > 10:
        print(f"\n  ... and {len(job_indices) - 10} more jobs")


def explore_location(graph, mappings, location_idx: int):
    """Explore a specific location and job distribution"""
    location_name = graph["location"].location_names[location_idx]

    print("\n" + "=" * 70)
    print(f"EXPLORING LOCATION: {location_name}")
    print("=" * 70)

    # Find all jobs in this location
    edge_index = graph[("location", "has", "job")].edge_index
    job_indices = edge_index[1, edge_index[0] == location_idx].tolist()

    print("\n📍 LOCATION STATISTICS:")
    print(f"  Total jobs: {len(job_indices)}")

    # Load job data for analysis
    df = pd.read_csv(f"{config.PROCESSED_DATA_PATH}jobs_processed.csv")

    # Salary distribution
    salaries = []
    for job_idx in job_indices:
        job_id = graph["job"].job_ids[job_idx]
        job_data = df[df["JobID"] == job_id].iloc[0]
        if job_data["salary_max"] > 0:
            salaries.append(job_data["salary_max"])

    if salaries:
        print("\n💰 SALARY STATISTICS:")
        print(f"  Average: {np.mean(salaries):.1f} million VND")
        print(f"  Min: {np.min(salaries):.1f} million VND")
        print(f"  Max: {np.max(salaries):.1f} million VND")


def main():
    """Main demo function"""
    print("\n" + "=" * 70)
    print(" " * 20 + "GRAPH EXPLORATION DEMO")
    print("=" * 70)

    # Load graph
    graph, mappings = load_graph()

    print("\n📊 GRAPH OVERVIEW:")
    print(f"  Jobs: {len(graph['job'].job_ids)}")
    print(f"  Companies: {len(graph['company'].company_names)}")
    print(f"  Locations: {len(graph['location'].location_names)}")

    # Demo 1: Explore a random job
    print("\n" + "-" * 70)
    print("DEMO 1: Explore a Job")
    print("-" * 70)
    explore_job(graph, mappings, job_idx=0)

    # Demo 2: Explore a company with many jobs
    print("\n" + "-" * 70)
    print("DEMO 2: Explore a Company")
    print("-" * 70)

    # Find company with most jobs
    edge_index = graph[("company", "posts", "job")].edge_index
    company_counts = torch.bincount(edge_index[0])
    top_company_idx = company_counts.argmax().item()

    explore_company(graph, mappings, company_idx=top_company_idx)

    # Demo 3: Explore a location
    print("\n" + "-" * 70)
    print("DEMO 3: Explore a Location")
    print("-" * 70)

    # Find location with most jobs
    edge_index = graph[("location", "has", "job")].edge_index
    location_counts = torch.bincount(edge_index[0])
    top_location_idx = location_counts.argmax().item()

    explore_location(graph, mappings, location_idx=top_location_idx)

    print("\n" + "=" * 70)
    print(" " * 25 + "DEMO COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
