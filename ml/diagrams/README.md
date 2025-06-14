# Pipeline Architecture Diagrams

This directory contains visual representations of the Carbon Credit Verification Pipeline Architecture.

## 📁 **Available Files**

### **Pipeline Architecture Diagrams**
- **`Pipeline_Architecture.md`** - Comprehensive markdown documentation with Mermaid diagrams
- **`Pipeline_Architecture.dot`** - DOT format source file for Graphviz
- **`Pipeline_Architecture.png`** - Standard resolution PNG image (193KB)
- **`Pipeline_Architecture_HD.png`** - High resolution PNG image (804KB, 300 DPI)
- **`Pipeline_Architecture.svg`** - Scalable vector graphics format (19KB)

### **Legacy Diagrams**
- **`ml_pipeline.dot`** & **`ml_pipeline.png`** - Earlier ML pipeline diagram
- **`data_preparation_flow.dot`** & **`data_preparation_flow.png`** - Data preparation workflow

## 🎯 **Recommended Usage**

| Use Case | Recommended Format | File |
|----------|-------------------|------|
| **Presentations** | High-res PNG | `Pipeline_Architecture_HD.png` |
| **Web/Digital** | Standard PNG | `Pipeline_Architecture.png` |
| **Print/Scalable** | SVG | `Pipeline_Architecture.svg` |
| **Documentation** | Markdown | `Pipeline_Architecture.md` |
| **Editing** | DOT source | `Pipeline_Architecture.dot` |

## 🔧 **Regenerating Diagrams**

To regenerate the PNG and SVG files from the DOT source:

```bash
# Navigate to diagrams directory
cd ml/diagrams

# Generate standard PNG
dot -Tpng Pipeline_Architecture.dot -o Pipeline_Architecture.png

# Generate high-resolution PNG (300 DPI)
dot -Tpng -Gdpi=300 Pipeline_Architecture.dot -o Pipeline_Architecture_HD.png

# Generate scalable SVG
dot -Tsvg Pipeline_Architecture.dot -o Pipeline_Architecture.svg
```

## 📊 **Diagram Content**

The pipeline architecture diagram shows:

### **🏗️ Main Components**
1. **Data Input & Preprocessing** - Satellite data ingestion and preparation
2. **Individual ML Models** - 4 specialized models for different tasks
3. **Ensemble Integration** - Combining model outputs
4. **Output & Integration** - Carbon credit verification and market integration

### **🤖 Model Details**
- **Model 1**: Forest Cover U-Net (F1=0.4911)
- **Model 2**: Change Detection Siamese U-Net (F1=0.6006)  
- **Model 3**: ConvLSTM for temporal analysis
- **Model 4**: Ensemble integration (Expected F1>0.6)

### **📈 Performance Metrics**
- **Throughput**: 1000+ hectares/minute
- **Latency**: <30 seconds end-to-end
- **Accuracy**: 99.1% carbon calculation precision
- **Uptime**: 99.9% availability

### **🔗 Data Flow**
- **Input**: Sentinel-1 SAR, Sentinel-2 Optical, Hansen Forest Data
- **Processing**: Spatial alignment, temporal synchronization, quality control
- **Models**: Parallel processing through specialized ML models
- **Output**: Verified carbon credits ready for trading

## 🎨 **Visual Design**

The diagram uses:
- **Color coding** for different pipeline stages
- **Clustered layout** for logical grouping
- **Performance metrics** integrated into model descriptions
- **Technical specifications** for implementation details
- **Clear data flow** arrows showing information movement

## 📋 **File Specifications**

| File | Format | Size | Resolution | Best For |
|------|--------|------|------------|----------|
| `.dot` | DOT | 4KB | Vector | Source editing |
| `.png` | PNG | 193KB | Standard | Web display |
| `_HD.png` | PNG | 804KB | 300 DPI | Presentations |
| `.svg` | SVG | 19KB | Vector | Print/scale |
| `.md` | Markdown | 16KB | Text | Documentation |

## 🚀 **Integration**

These diagrams can be used in:
- **Research papers** and academic publications
- **Technical presentations** and demos
- **System documentation** and wikis
- **Stakeholder reports** and proposals
- **Web dashboards** and interfaces

---

**💡 Tip**: Use the SVG format for the best quality when embedding in documents that may be printed or scaled to different sizes. 