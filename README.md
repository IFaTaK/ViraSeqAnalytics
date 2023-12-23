# ViraSeqAnalytics

![AI-Generated Banner](assets/project-banner.png)  
*Image above: AI-generated banner representing RNA sequencing, virus variant detection, and data analysis.*

## Overview
This project focuses on the identification of coronavirus variants through advanced genomic analysis. It is a part of my academic work, specifically a TIPE (a French academic initiative for hands-on, research-oriented projects), emphasizing personal initiative and research. The aim is to utilize shotgun sequencing, the Burrows-Wheeler Transform (BWT), FM-index, and principal component analysis to analyze and interpret genomic data related to COVID-19.

## Objective
The main goal is to identify specific variants by comparing the genome to a reference sequence. We first reconstruct the full coronavirus genome, which is fragmented into several tiny `reads`. Then, we combine BWT and FM-index for data extraction and principal component analysis for in-depth genomic analysis.

## Methodology
1. **Shotgun Sequencing**: Employing shotgun sequencing for fast and efficient reading of coronavirus genome fragments.
2. **Genome Reconstruction**: Using complex algorithms and optimization to reconstruct the full genome.
3. **Data Extraction**: Using BWT and FM-index for comparing fragmented genomic data with reference genomes.
4. **Data Analysis**: Implementing principal component analysis to deeply analyze the genomic data.
5. **Variant Identification**: Using clustering algorithms to classify different variants.

## Tools and Technologies
- **Shotgun Sequencing**
- **BWT & FM-index**
- **Principal Component Analysis**
- **K-means Clustering**
- **Programming Language**: Python

## Data Source
- Genomic data for this project was sourced from the [NCBI website](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/#/virus?SeqType_s=Nucleotide&VirusLineage_ss=Severe%20acute%20respiratory%20syndrome%20coronavirus%202,%20taxid:2697049).

## Current Status and Future Enhancements
- Initial studies have successfully identified two geographic variants of the coronavirus collected in the same period.
- Future plans include expanding data collection using the NCBI API, enhancing FASTA file reading capabilities, and developing methods for cluster detection and variant grouping.

## Installation and Usage
(Provide instructions on how to install and use your project.)

## Acknowledgements
This project was initiated as a collaborative effort between [Kilian LEFEVRE](https://github.com/IFaTaK), Anthony CANNIAUX, and Baptiste ORTILLION. Kilian LEFEVRE is currently leading further development and enhancements.

## License
This project is licensed under the GNU General Public License.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes. Adhere to the existing code style and include tests for new features.

## Contact

For any inquiries or further information about this project, please use the [Issues section of this GitHub repository](https://github.com/IFaTaK/RNA_Sequencing/issues). This is the preferred channel for questions, suggestions, or discussions regarding the project.

For more about my work and contributions, you can visit [my GitHub profile](https://github.com/IFaTaK).