"""
Example script demonstrating Clay foundation model embeddings with segment-geospatial.

This script shows how to:
1. Load a geospatial image
2. Generate Clay foundation model embeddings
3. Save and load embeddings
4. Visualize embedding results

Requirements:
- Clay model checkpoint file
- Geospatial imagery (GeoTIFF, etc.)
- Clay model dependencies: claymodel, torch, torchvision, pyyaml, python-box
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from samgeo import Clay, load_embeddings


def main():
    # Configuration
    CHECKPOINT_PATH = "path/to/clay-model-checkpoint.ckpt"  # Update this path
    IMAGE_PATH = "path/to/your/satellite_image.tif"  # Update this path
    OUTPUT_DIR = "clay_embeddings_output"

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Clay Foundation Model Embeddings Example ===\n")

    # Step 1: Initialize Clay embeddings model
    print("1. Initializing Clay model...")
    try:
        clay = Clay(
            checkpoint_path=CHECKPOINT_PATH,
            device="auto",  # Will use GPU if available
            mask_ratio=0.0,  # No masking for inference
            shuffle=False,
        )
        print("   ✓ Clay model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading Clay model: {e}")
        print("   Please ensure you have:")
        print("   - Valid Clay checkpoint file")
        print(
            "   - Clay dependencies: pip install claymodel torch torchvision pyyaml python-box"
        )
        return

    # Step 2: Load and analyze image
    print("\n2. Loading geospatial image...")
    try:
        clay.set_image(
            source=IMAGE_PATH,
            # sensor_type="sentinel-2-l2a",  # Optional: override auto-detection
            # date="2023-06-01",             # Optional: specify acquisition date
            # gsd_override=10.0              # Optional: override ground sample distance
        )
        print("   ✓ Image loaded and analyzed")
        print(f"   - Image shape: {clay.image.shape}")
        print(f"   - Detected sensor: {clay.sensor_type}")
        print(f"   - Center coordinates: ({clay.lat:.4f}, {clay.lon:.4f})")
    except Exception as e:
        print(f"   ✗ Error loading image: {e}")
        print("   Please check the image path and format")
        return

    # Step 3: Generate embeddings
    print("\n3. Generating Clay embeddings...")
    try:
        # For large images, process in tiles
        embeddings_result = clay.generate_embeddings(
            tile_size=256,  # Size of processing tiles
            overlap=0.1,  # 10% overlap between tiles
        )

        print("   ✓ Embeddings generated successfully")
        print(f"   - Number of tiles: {embeddings_result['num_tiles']}")
        print(f"   - Embedding shape: {embeddings_result['embeddings'].shape}")
        print(f"   - Feature dimension: {embeddings_result['embeddings'].shape[-1]}")

    except Exception as e:
        print(f"   ✗ Error generating embeddings: {e}")
        return

    # Step 4: Save embeddings
    print("\n4. Saving embeddings...")
    try:
        embeddings_file = os.path.join(OUTPUT_DIR, "clay_embeddings.npz")
        clay.save_embeddings(embeddings_result, embeddings_file, format="npz")
        print(f"   ✓ Embeddings saved to {embeddings_file}")
    except Exception as e:
        print(f"   ✗ Error saving embeddings: {e}")
        return

    # Step 5: Load and verify embeddings
    print("\n5. Loading and verifying saved embeddings...")
    try:
        loaded_embeddings = load_embeddings(embeddings_file)
        print("   ✓ Embeddings loaded successfully")
        print(f"   - Sensor type: {loaded_embeddings['sensor_type']}")
        print(f"   - Number of tiles: {loaded_embeddings['num_tiles']}")
        print(f"   - Original image shape: {loaded_embeddings['image_shape']}")
    except Exception as e:
        print(f"   ✗ Error loading embeddings: {e}")
        return

    # Step 6: Visualize results
    print("\n6. Creating visualizations...")
    try:
        # Plot RGB image if available
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Original image (RGB bands if available)
        image = clay.image
        if clay.sensor_type in clay.metadata:
            rgb_indices = clay.metadata[clay.sensor_type].get("rgb_indices", [0, 1, 2])
            if len(rgb_indices) == 3 and image.shape[2] >= max(rgb_indices) + 1:
                rgb_image = image[:, :, rgb_indices]
                # Normalize for display
                rgb_image = np.clip(rgb_image / np.percentile(rgb_image, 98), 0, 1)
                axes[0].imshow(rgb_image)
                axes[0].set_title(f"Original Image ({clay.sensor_type})")
                axes[0].axis("off")
            else:
                axes[0].imshow(image[:, :, 0], cmap="gray")
                axes[0].set_title("Original Image (First Band)")
                axes[0].axis("off")
        else:
            axes[0].imshow(image[:, :, 0], cmap="gray")
            axes[0].set_title("Original Image (First Band)")
            axes[0].axis("off")

        # Embedding visualization (PCA of first tile)
        embeddings = embeddings_result["embeddings"]
        if embeddings.shape[0] > 0:
            # Use first embedding for visualization
            first_embedding = embeddings[0].flatten()

            # Create a simple visualization of embedding values
            embedding_2d = first_embedding[:256].reshape(
                16, 16
            )  # Take first 256 values
            axes[1].imshow(embedding_2d, cmap="viridis")
            axes[1].set_title(
                "Clay Embedding Visualization\n(First 256 features, first tile)"
            )
            axes[1].axis("off")

        plt.tight_layout()

        # Save plot
        plot_file = os.path.join(OUTPUT_DIR, "clay_embeddings_visualization.png")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"   ✓ Visualization saved to {plot_file}")

    except Exception as e:
        print(f"   ✗ Error creating visualizations: {e}")

    # Step 7: Demonstrate embedding analysis
    print("\n7. Embedding analysis...")
    try:
        embeddings = embeddings_result["embeddings"]

        # Basic statistics
        print(f"   - Embedding statistics:")
        print(f"     * Mean: {np.mean(embeddings):.4f}")
        print(f"     * Std:  {np.std(embeddings):.4f}")
        print(f"     * Min:  {np.min(embeddings):.4f}")
        print(f"     * Max:  {np.max(embeddings):.4f}")

        # Similarity between tiles (if multiple tiles)
        if embeddings.shape[0] > 1:
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity(embeddings)
            avg_similarity = np.mean(
                similarities[np.triu_indices_from(similarities, k=1)]
            )
            print(f"     * Average tile similarity: {avg_similarity:.4f}")

        print("   ✓ Analysis complete")

    except Exception as e:
        print(f"   ✗ Error in embedding analysis: {e}")

    print(f"\n=== Example completed successfully! ===")
    print(f"Output files saved in: {OUTPUT_DIR}/")
    print("\nNext steps:")
    print("- Use embeddings for similarity search")
    print("- Fine-tune on downstream tasks")
    print("- Integrate with SAM for enhanced segmentation")


def example_with_numpy_array():
    """Example showing how to use Clay embeddings with numpy arrays."""
    print("\n=== Numpy Array Example ===")

    # Create a synthetic 4-band image (RGBI)
    synthetic_image = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)

    try:
        # Initialize Clay model
        clay = ClayEmbeddings(
            checkpoint_path="path/to/clay-model-checkpoint.ckpt", device="auto"
        )

        # Set synthetic image
        clay.set_image(
            source=synthetic_image,
            sensor_type="naip",  # Specify sensor type for numpy arrays
            date="2023-06-01",
        )

        # Generate embeddings
        result = clay.generate_embeddings(tile_size=256)

        print(f"Generated embeddings for synthetic image:")
        print(f"- Shape: {result['embeddings'].shape}")
        print(f"- Sensor: {result['sensor_type']}")

    except Exception as e:
        print(f"Error in numpy array example: {e}")


if __name__ == "__main__":
    main()

    # Uncomment to run numpy array example
    # example_with_numpy_array()
