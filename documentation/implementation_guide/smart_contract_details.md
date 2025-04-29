## Smart Contract Details for Carbon Credit Verification SaaS

This section details the design, implementation, and security considerations for the Solidity smart contract used to issue and manage carbon credit verification certificates on the Polygon blockchain.

### 1. Purpose and Core Functionality

-   **Purpose**: To create immutable, transparent, and verifiable records of carbon credit verification results on the blockchain.
-   **Standard**: We will use the ERC-721 Non-Fungible Token (NFT) standard. Each NFT will represent a unique verification certificate for a specific project and verification period.
-   **Core Functions**:
    -   Minting (Issuing) new certificate NFTs upon successful verification.
    -   Storing essential verification metadata (or a link to it).
    -   Querying certificate details.
    -   (Optional) Transferring ownership of the certificate/credit.
    -   (Optional) Retiring the credit to prevent double counting.

### 2. Data Stored On-Chain

Each ERC-721 token (certificate) will store:

-   **Token ID**: Unique identifier for the NFT (can correspond to the off-chain `verification_id`).
-   **Owner**: The address owning the certificate (initially the project owner or the platform).
-   **Token URI**: A URL pointing to off-chain metadata (JSON file stored on IPFS or a persistent web server). This metadata file will contain detailed information.

**Metadata JSON (Stored Off-Chain, linked via Token URI)**:

```json
{
  "name": "Carbon Verification Certificate #123",
  "description": "Verified carbon sequestration for Project XYZ (ID: 45) for the period 2022-01-01 to 2022-12-31.",
  "image": "ipfs://<cid_of_certificate_visual_representation>", // Optional visual
  "external_url": "https://your-saas.com/verifications/123", // Link back to the SaaS platform
  "attributes": [
    { "trait_type": "Verification ID", "value": 123 },
    { "trait_type": "Project ID", "value": 45 },
    { "trait_type": "Project Name", "value": "Project XYZ Reforestation" },
    { "trait_type": "Verification Period Start", "value": "2022-01-01" },
    { "trait_type": "Verification Period End", "value": "2022-12-31" },
    { "trait_type": "Verified Carbon Impact (tCO2e)", "value": 5432.1 }, 
    { "trait_type": "Verification Standard", "value": "Internal AI + Human Review" }, // Or VCS, Gold Standard etc.
    { "trait_type": "Verification Date", "value": "2023-03-15" },
    { "trait_type": "Status", "value": "Issued" } // Could be updated to "Retired"
  ]
}
```

**Rationale for Off-Chain Metadata**: Storing large amounts of data directly on-chain is expensive. Linking to metadata via IPFS ensures data persistence and immutability while keeping gas costs manageable.

### 3. Smart Contract Design (Solidity - ERC721)

We will use OpenZeppelin's ERC721 implementation for robustness and security.

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol"; // Or AccessControl for more granular roles
import "@openzeppelin/contracts/utils/Counters.sol";

contract CarbonVerificationCertificate is ERC721, ERC721URIStorage, Ownable {
    using Counters for Counters.Counter;
    Counters.Counter private _tokenIdCounter; // To generate unique token IDs

    // Mapping from Verification ID (off-chain) to Token ID (on-chain)
    mapping(uint256 => uint256) private _verificationIdToTokenId;
    mapping(uint256 => bool) private _verificationIdExists;

    // Mapping to track retired tokens
    mapping(uint256 => bool) private _retiredTokens;

    // Event emitted when a certificate is issued
    event CertificateIssued(
        uint256 indexed tokenId,
        uint256 indexed verificationId,
        address indexed owner,
        string tokenURI,
        uint256 timestamp
    );

    // Event emitted when a certificate is retired
    event CertificateRetired(
        uint256 indexed tokenId,
        address indexed retiredBy,
        uint256 timestamp
    );

    constructor() ERC721("Carbon Verification Certificate", "CVC") {}

    /**
     * @dev Issues a new verification certificate NFT.
     * Only callable by the contract owner (or an authorized minter role).
     * @param ownerAddress The address to receive the certificate NFT.
     * @param verificationId The unique ID from the off-chain verification system.
     * @param tokenURI_ The URI pointing to the certificate's metadata (e.g., IPFS link).
     */
    function issueCertificate(
        address ownerAddress,
        uint256 verificationId,
        string memory tokenURI_
    ) public onlyOwner { // Use onlyOwner or a specific minter role
        require(ownerAddress != address(0), "ERC721: mint to the zero address");
        require(!_verificationIdExists[verificationId], "Certificate already issued for this verification ID");

        _tokenIdCounter.increment();
        uint256 newTokenId = _tokenIdCounter.current();

        _safeMint(ownerAddress, newTokenId);
        _setTokenURI(newTokenId, tokenURI_);

        _verificationIdToTokenId[verificationId] = newTokenId;
        _verificationIdExists[verificationId] = true;

        emit CertificateIssued(newTokenId, verificationId, ownerAddress, tokenURI_, block.timestamp);
    }

    /**
     * @dev Retires a certificate NFT, preventing further transfers.
     * Can only be called by the current owner of the NFT.
     * @param tokenId The ID of the token to retire.
     */
    function retireCertificate(uint256 tokenId) public {
        require(_exists(tokenId), "Certificate does not exist");
        require(ownerOf(tokenId) == msg.sender, "Only the owner can retire the certificate");
        require(!_retiredTokens[tokenId], "Certificate already retired");

        _retiredTokens[tokenId] = true;

        // Optional: Burn the token after retiring to remove it completely
        // _burn(tokenId);

        emit CertificateRetired(tokenId, msg.sender, block.timestamp);
    }

    /**
     * @dev Checks if a token is retired.
     */
    function isRetired(uint256 tokenId) public view returns (bool) {
        return _retiredTokens[tokenId];
    }

    /**
     * @dev Gets the token ID associated with an off-chain verification ID.
     */
    function getTokenIdByVerificationId(uint256 verificationId) public view returns (uint256) {
        require(_verificationIdExists[verificationId], "No certificate found for this verification ID");
        return _verificationIdToTokenId[verificationId];
    }

    // --- Overrides for ERC721 and ERC721URIStorage --- 

    /**
     * @dev Override _beforeTokenTransfer to prevent transfer of retired tokens.
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId
    ) internal virtual override {
        super._beforeTokenTransfer(from, to, tokenId);
        require(!_retiredTokens[tokenId], "Cannot transfer a retired certificate");
    }

    /**
     * @dev See {IERC721Metadata-tokenURI}.
     */
    function tokenURI(uint256 tokenId) 
        public 
        view 
        virtual 
        override(ERC721, ERC721URIStorage) 
        returns (string memory) 
    {
        return super.tokenURI(tokenId);
    }

    /**
     * @dev See {IERC165-supportsInterface}.
     */
    function supportsInterface(bytes4 interfaceId) 
        public 
        view 
        virtual 
        override(ERC721, ERC721URIStorage) 
        returns (bool) 
    {
        return super.supportsInterface(interfaceId);
    }

    /**
     * @dev Override _burn to also clean up retirement status (if needed) and verification ID mapping.
     * Note: Burning might not be desired if historical record needs to be fully preserved.
     * Consider if retiring without burning is sufficient.
     */
    function _burn(uint256 tokenId) internal virtual override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
        // If burning, potentially clean up mappings, though leaving them might be better for history.
        // delete _retiredTokens[tokenId]; 
        // Consider how to handle verificationId mapping if burning.
    }
}
```

### 4. Key Functions Explained

-   **`constructor()`**: Initializes the ERC721 token with a name and symbol.
-   **`issueCertificate(address ownerAddress, uint256 verificationId, string memory tokenURI_)`**: Mints a new NFT. Only callable by the `owner` (deployer) or a designated minter role. It checks if a certificate for the `verificationId` already exists, mints the token, sets its metadata URI, and maps the `verificationId` to the new `tokenId`. Emits a `CertificateIssued` event.
-   **`retireCertificate(uint256 tokenId)`**: Marks a token as retired. Can only be called by the token's current owner. Prevents further transfers. Emits a `CertificateRetired` event.
-   **`isRetired(uint256 tokenId)`**: Public view function to check the retirement status of a token.
-   **`getTokenIdByVerificationId(uint256 verificationId)`**: Allows querying the on-chain `tokenId` using the off-chain `verificationId`.
-   **`_beforeTokenTransfer(...)`**: Hook overridden from OpenZeppelin's ERC721 contract to add the check preventing transfers of retired tokens.
-   **`tokenURI(...)`**, **`supportsInterface(...)`**, **`_burn(...)`**: Overrides required by inheriting from multiple OpenZeppelin contracts.

### 5. Security Considerations

-   **Access Control**: The `issueCertificate` function is critical. Using `Ownable` is simple, but OpenZeppelin's `AccessControl` is more flexible, allowing multiple minter roles without giving full ownership.
-   **Input Validation**: The `issueCertificate` function includes basic checks (`ownerAddress != address(0)`, `!_verificationIdExists`). Add more checks as needed.
-   **Reentrancy**: The use of standard OpenZeppelin contracts provides protection against basic reentrancy attacks. Avoid complex external calls within state-changing functions.
-   **Gas Limits**: Be mindful of gas costs, especially if iterating over data structures. The current design avoids loops in critical functions.
-   **Metadata Immutability**: Ensure the `tokenURI` points to immutable storage like IPFS. If using a centralized server, the metadata can be changed, reducing trust.
-   **Upgradability**: Consider using an upgradeable contract pattern (e.g., OpenZeppelin Upgrades Plugins with proxies) if you anticipate needing to modify the contract logic after deployment without losing state.
-   **Testing**: Thoroughly test the contract on testnets (like Polygon Mumbai) using frameworks like Hardhat or Truffle. Write unit tests covering all functions and edge cases.
-   **Auditing**: For a production deployment, obtain a professional security audit of the smart contract code.

### 6. Deployment and Verification

-   **Tools**: Hardhat, Truffle, Remix IDE.
-   **Network**: Deploy initially to Polygon Mumbai testnet for thorough testing.
-   **Deployment Script**: Use Hardhat/Truffle scripts to manage deployment, passing constructor arguments if any.
-   **Verification**: Verify the contract source code on PolygonScan (Mumbai and Mainnet) to provide transparency and allow users to interact with it directly.
-   **Backend Integration**: Store the deployed contract address and ABI in the backend configuration (`settings.CONTRACT_ADDRESS`, `settings.CONTRACT_ABI_PATH`). The backend service (`BlockchainService`) will use these to interact with the deployed contract.

This smart contract design provides a secure and standardized way to represent carbon verification certificates as NFTs on the Polygon blockchain, integrating seamlessly with the off-chain SaaS application.
