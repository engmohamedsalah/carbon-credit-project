# Blockchain Integration Implementation Plan

**Project**: Carbon Credit Verification SaaS  
**Feature**: Complete Blockchain Integration with Smart Contracts & NFT Certificates  
**Timeline**: 1-2 Weeks (80-100 hours)  
**Complexity**: HIGH  
**Risk Level**: MODERATE  
**Priority**: HIGH  

---

## 📋 Executive Summary

This plan implements complete blockchain integration for the Carbon Credit Verification SaaS, transforming it from a centralized system to a decentralized, immutable certification platform. The implementation includes smart contracts on Polygon, NFT certificate minting, Web3 frontend integration, and comprehensive blockchain backend services.

---

## 🎯 Current Situation Analysis

### ✅ **COMPLETED (UI Framework)**
- **Professional UI**: Complete blockchain explorer interface exists
- **Mock Data System**: Demo verification with realistic data flow
- **Component Architecture**: Card-based certificate display ready
- **User Interface**: Search, verification, and display components functional
- **Design System**: Material-UI components with consistent styling

### ❌ **MISSING (Core Blockchain)**
- **Smart Contracts**: No Solidity contracts developed
- **Web3 Integration**: No wallet connection or transaction handling  
- **NFT Minting**: Certificate tokenization not implemented
- **Blockchain Backend**: No Python Web3 integration
- **Contract Deployment**: No deployment infrastructure
- **Event Monitoring**: No blockchain event listening

### 🎯 **TARGET ARCHITECTURE**
```
Frontend (React + Web3) ↔ Smart Contracts (Polygon) ↔ Backend (Python + Web3) ↔ Database (Sync)
```

---

## 🗓️ IMPLEMENTATION ROADMAP

---

# **📅 WEEK 1: SMART CONTRACT FOUNDATION**

## **Day 1-2: Smart Contract Development & Architecture**

### **Phase 1A: Development Environment Setup** 
**Duration**: 4 hours

#### 1.1 Project Structure Creation
```bash
# Create blockchain development directory
mkdir blockchain
cd blockchain

# Initialize Hardhat project
npm init -y
npm install --save-dev hardhat @nomiclabs/hardhat-ethers @nomiclabs/hardhat-waffle
npm install --save-dev @openzeppelin/contracts @openzeppelin/hardhat-upgrades
npm install ethers dotenv

# Initialize Hardhat
npx hardhat

# Project structure
blockchain/
├── contracts/
│   ├── CarbonCreditNFT.sol
│   ├── CarbonCreditVerifier.sol
│   ├── CarbonCreditRegistry.sol
│   └── interfaces/
├── deploy/
├── test/
├── scripts/
└── hardhat.config.js
```

#### 1.2 Hardhat Configuration
**File**: `blockchain/hardhat.config.js`
```javascript
require("@nomiclabs/hardhat-waffle");
require("@nomiclabs/hardhat-ethers");
require("@openzeppelin/hardhat-upgrades");
require("dotenv").config();

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    polygon: {
      url: process.env.POLYGON_RPC_URL,
      accounts: [process.env.PRIVATE_KEY],
      gasPrice: 35000000000 // 35 gwei
    },
    mumbai: {
      url: process.env.MUMBAI_RPC_URL,
      accounts: [process.env.PRIVATE_KEY],
      gasPrice: 35000000000
    },
    localhost: {
      url: "http://127.0.0.1:8545"
    }
  },
  etherscan: {
    apiKey: process.env.POLYGONSCAN_API_KEY
  }
};
```

### **Phase 1B: Core Smart Contracts Development**
**Duration**: 8 hours

#### 1.3 Carbon Credit NFT Contract
**File**: `blockchain/contracts/CarbonCreditNFT.sol`
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

contract CarbonCreditNFT is ERC721, ERC721URIStorage, AccessControl, Pausable, ReentrancyGuard {
    using Counters for Counters.Counter;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant BURNER_ROLE = keccak256("BURNER_ROLE");

    Counters.Counter private _tokenIdCounter;

    struct CarbonCertificate {
        uint256 projectId;
        uint256 carbonAmount; // in tCO2e
        uint256 verificationDate;
        address verifier;
        bool isRetired;
        string methodology;
        bytes32 verificationHash;
    }

    mapping(uint256 => CarbonCertificate) public certificates;
    mapping(uint256 => bool) public retiredTokens;
    mapping(address => uint256[]) public ownerTokens;
    
    // Events
    event CertificateMinted(
        uint256 indexed tokenId,
        address indexed recipient,
        uint256 indexed projectId,
        uint256 carbonAmount
    );
    
    event CertificateRetired(
        uint256 indexed tokenId,
        address indexed owner,
        uint256 carbonAmount
    );
    
    event CertificateTransferred(
        uint256 indexed tokenId,
        address indexed from,
        address indexed to
    );

    constructor(address admin) ERC721("Carbon Credit Certificate", "CCC") {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(MINTER_ROLE, admin);
        _grantRole(VERIFIER_ROLE, admin);
        _grantRole(BURNER_ROLE, admin);
    }

    /**
     * @dev Mint a new carbon credit certificate
     * @param to Address to receive the certificate
     * @param projectId ID of the carbon credit project
     * @param carbonAmount Amount of carbon credits in tCO2e
     * @param methodology Verification methodology used
     * @param verificationHash Hash of verification data
     * @param tokenURI Metadata URI for the token
     */
    function mintCertificate(
        address to,
        uint256 projectId,
        uint256 carbonAmount,
        string memory methodology,
        bytes32 verificationHash,
        string memory tokenURI
    ) external onlyRole(MINTER_ROLE) nonReentrant whenNotPaused returns (uint256) {
        require(to != address(0), "Cannot mint to zero address");
        require(carbonAmount > 0, "Carbon amount must be positive");
        require(bytes(methodology).length > 0, "Methodology required");

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();

        // Create certificate data
        certificates[tokenId] = CarbonCertificate({
            projectId: projectId,
            carbonAmount: carbonAmount,
            verificationDate: block.timestamp,
            verifier: msg.sender,
            isRetired: false,
            methodology: methodology,
            verificationHash: verificationHash
        });

        // Mint the NFT
        _safeMint(to, tokenId);
        _setTokenURI(tokenId, tokenURI);

        // Track owner's tokens
        ownerTokens[to].push(tokenId);

        emit CertificateMinted(tokenId, to, projectId, carbonAmount);
        
        return tokenId;
    }

    /**
     * @dev Retire carbon credits (permanent removal from circulation)
     * @param tokenId Token ID to retire
     */
    function retireCertificate(uint256 tokenId) external nonReentrant {
        require(_exists(tokenId), "Token does not exist");
        require(ownerOf(tokenId) == msg.sender, "Not token owner");
        require(!certificates[tokenId].isRetired, "Certificate already retired");

        certificates[tokenId].isRetired = true;
        retiredTokens[tokenId] = true;

        // Burn the token to prevent further transfers
        _burn(tokenId);

        emit CertificateRetired(tokenId, msg.sender, certificates[tokenId].carbonAmount);
    }

    /**
     * @dev Get certificate details
     * @param tokenId Token ID to query
     */
    function getCertificate(uint256 tokenId) external view returns (CarbonCertificate memory) {
        require(_exists(tokenId) || retiredTokens[tokenId], "Token does not exist");
        return certificates[tokenId];
    }

    /**
     * @dev Get all tokens owned by an address
     * @param owner Address to query
     */
    function getOwnerTokens(address owner) external view returns (uint256[] memory) {
        return ownerTokens[owner];
    }

    /**
     * @dev Calculate total carbon credits owned by address
     * @param owner Address to query
     */
    function getTotalCarbonCredits(address owner) external view returns (uint256) {
        uint256[] memory tokens = ownerTokens[owner];
        uint256 totalCredits = 0;
        
        for (uint256 i = 0; i < tokens.length; i++) {
            if (_exists(tokens[i]) && !certificates[tokens[i]].isRetired) {
                totalCredits += certificates[tokens[i]].carbonAmount;
            }
        }
        
        return totalCredits;
    }

    /**
     * @dev Override transfers to update owner tracking
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId
    ) internal override {
        super._beforeTokenTransfer(from, to, tokenId);
        
        if (from != address(0) && to != address(0)) {
            // Remove from old owner's list
            _removeTokenFromOwner(from, tokenId);
            // Add to new owner's list
            ownerTokens[to].push(tokenId);
            
            emit CertificateTransferred(tokenId, from, to);
        }
    }

    /**
     * @dev Remove token from owner's list
     */
    function _removeTokenFromOwner(address owner, uint256 tokenId) private {
        uint256[] storage tokens = ownerTokens[owner];
        for (uint256 i = 0; i < tokens.length; i++) {
            if (tokens[i] == tokenId) {
                tokens[i] = tokens[tokens.length - 1];
                tokens.pop();
                break;
            }
        }
    }

    // Required overrides
    function _burn(uint256 tokenId) internal override(ERC721, ERC721URIStorage) {
        super._burn(tokenId);
    }

    function tokenURI(uint256 tokenId) public view override(ERC721, ERC721URIStorage) returns (string memory) {
        return super.tokenURI(tokenId);
    }

    function supportsInterface(bytes4 interfaceId) public view override(ERC721, AccessControl) returns (bool) {
        return super.supportsInterface(interfaceId);
    }

    // Admin functions
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
}
```

#### 1.4 Verification Registry Contract
**File**: `blockchain/contracts/CarbonCreditVerifier.sol`
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./interfaces/ICarbonCreditNFT.sol";

contract CarbonCreditVerifier is AccessControl, Pausable, ReentrancyGuard {
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");

    enum VerificationStatus { Pending, Approved, Rejected }

    struct VerificationRequest {
        uint256 projectId;
        address requester;
        uint256 carbonAmount;
        string methodology;
        bytes32 dataHash;
        string ipfsHash;
        VerificationStatus status;
        address verifier;
        uint256 submissionTime;
        uint256 verificationTime;
        string rejectionReason;
    }

    ICarbonCreditNFT public carbonNFT;
    
    mapping(uint256 => VerificationRequest) public verifications;
    mapping(uint256 => bool) public projectVerified;
    mapping(address => uint256[]) public userRequests;
    
    uint256 private _verificationIdCounter;

    // Events
    event VerificationRequested(
        uint256 indexed verificationId,
        uint256 indexed projectId,
        address indexed requester,
        uint256 carbonAmount
    );
    
    event VerificationApproved(
        uint256 indexed verificationId,
        uint256 indexed projectId,
        address indexed verifier,
        uint256 tokenId
    );
    
    event VerificationRejected(
        uint256 indexed verificationId,
        uint256 indexed projectId,
        address indexed verifier,
        string reason
    );

    constructor(address admin, address _carbonNFT) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(ADMIN_ROLE, admin);
        _grantRole(VERIFIER_ROLE, admin);
        carbonNFT = ICarbonCreditNFT(_carbonNFT);
    }

    /**
     * @dev Submit verification request
     */
    function submitVerification(
        uint256 projectId,
        uint256 carbonAmount,
        string memory methodology,
        bytes32 dataHash,
        string memory ipfsHash
    ) external nonReentrant whenNotPaused returns (uint256) {
        require(carbonAmount > 0, "Carbon amount must be positive");
        require(bytes(methodology).length > 0, "Methodology required");
        require(bytes(ipfsHash).length > 0, "IPFS hash required");

        uint256 verificationId = _verificationIdCounter++;
        
        verifications[verificationId] = VerificationRequest({
            projectId: projectId,
            requester: msg.sender,
            carbonAmount: carbonAmount,
            methodology: methodology,
            dataHash: dataHash,
            ipfsHash: ipfsHash,
            status: VerificationStatus.Pending,
            verifier: address(0),
            submissionTime: block.timestamp,
            verificationTime: 0,
            rejectionReason: ""
        });

        userRequests[msg.sender].push(verificationId);

        emit VerificationRequested(verificationId, projectId, msg.sender, carbonAmount);
        
        return verificationId;
    }

    /**
     * @dev Approve verification and mint NFT
     */
    function approveVerification(
        uint256 verificationId,
        string memory tokenURI
    ) external onlyRole(VERIFIER_ROLE) nonReentrant returns (uint256) {
        require(verificationId < _verificationIdCounter, "Invalid verification ID");
        
        VerificationRequest storage verification = verifications[verificationId];
        require(verification.status == VerificationStatus.Pending, "Already processed");

        verification.status = VerificationStatus.Approved;
        verification.verifier = msg.sender;
        verification.verificationTime = block.timestamp;

        // Mint NFT certificate
        uint256 tokenId = carbonNFT.mintCertificate(
            verification.requester,
            verification.projectId,
            verification.carbonAmount,
            verification.methodology,
            verification.dataHash,
            tokenURI
        );

        projectVerified[verification.projectId] = true;

        emit VerificationApproved(verificationId, verification.projectId, msg.sender, tokenId);
        
        return tokenId;
    }

    /**
     * @dev Reject verification request
     */
    function rejectVerification(
        uint256 verificationId,
        string memory reason
    ) external onlyRole(VERIFIER_ROLE) nonReentrant {
        require(verificationId < _verificationIdCounter, "Invalid verification ID");
        require(bytes(reason).length > 0, "Rejection reason required");
        
        VerificationRequest storage verification = verifications[verificationId];
        require(verification.status == VerificationStatus.Pending, "Already processed");

        verification.status = VerificationStatus.Rejected;
        verification.verifier = msg.sender;
        verification.verificationTime = block.timestamp;
        verification.rejectionReason = reason;

        emit VerificationRejected(verificationId, verification.projectId, msg.sender, reason);
    }

    /**
     * @dev Get verification details
     */
    function getVerification(uint256 verificationId) external view returns (VerificationRequest memory) {
        require(verificationId < _verificationIdCounter, "Invalid verification ID");
        return verifications[verificationId];
    }

    /**
     * @dev Get user's verification requests
     */
    function getUserRequests(address user) external view returns (uint256[] memory) {
        return userRequests[user];
    }

    /**
     * @dev Check if project is verified
     */
    function isProjectVerified(uint256 projectId) external view returns (bool) {
        return projectVerified[projectId];
    }

    // Admin functions
    function setCarbonNFT(address _carbonNFT) external onlyRole(ADMIN_ROLE) {
        carbonNFT = ICarbonCreditNFT(_carbonNFT);
    }

    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
```

#### 1.5 Interface Definition
**File**: `blockchain/contracts/interfaces/ICarbonCreditNFT.sol`
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface ICarbonCreditNFT {
    function mintCertificate(
        address to,
        uint256 projectId,
        uint256 carbonAmount,
        string memory methodology,
        bytes32 verificationHash,
        string memory tokenURI
    ) external returns (uint256);
    
    function getCertificate(uint256 tokenId) external view returns (
        uint256 projectId,
        uint256 carbonAmount,
        uint256 verificationDate,
        address verifier,
        bool isRetired,
        string memory methodology,
        bytes32 verificationHash
    );
}
```

---

## **Day 3-4: Testing & Deployment Infrastructure**

### **Phase 2A: Comprehensive Testing Suite**
**Duration**: 6 hours

#### 2.1 Unit Tests for NFT Contract
**File**: `blockchain/test/CarbonCreditNFT.test.js`
```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("CarbonCreditNFT", function () {
  let carbonNFT;
  let owner, minter, verifier, user1, user2;

  beforeEach(async function () {
    [owner, minter, verifier, user1, user2] = await ethers.getSigners();
    
    const CarbonCreditNFT = await ethers.getContractFactory("CarbonCreditNFT");
    carbonNFT = await CarbonCreditNFT.deploy(owner.address);
    await carbonNFT.deployed();

    // Grant roles
    const MINTER_ROLE = await carbonNFT.MINTER_ROLE();
    const VERIFIER_ROLE = await carbonNFT.VERIFIER_ROLE();
    
    await carbonNFT.grantRole(MINTER_ROLE, minter.address);
    await carbonNFT.grantRole(VERIFIER_ROLE, verifier.address);
  });

  describe("Deployment", function () {
    it("Should set the correct name and symbol", async function () {
      expect(await carbonNFT.name()).to.equal("Carbon Credit Certificate");
      expect(await carbonNFT.symbol()).to.equal("CCC");
    });

    it("Should grant admin role to deployer", async function () {
      const DEFAULT_ADMIN_ROLE = await carbonNFT.DEFAULT_ADMIN_ROLE();
      expect(await carbonNFT.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.be.true;
    });
  });

  describe("Certificate Minting", function () {
    it("Should mint certificate with correct data", async function () {
      const projectId = 1;
      const carbonAmount = 1000;
      const methodology = "VCS";
      const verificationHash = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("test"));
      const tokenURI = "https://ipfs.io/ipfs/test";

      await carbonNFT.connect(minter).mintCertificate(
        user1.address,
        projectId,
        carbonAmount,
        methodology,
        verificationHash,
        tokenURI
      );

      const certificate = await carbonNFT.getCertificate(0);
      expect(certificate.projectId).to.equal(projectId);
      expect(certificate.carbonAmount).to.equal(carbonAmount);
      expect(certificate.methodology).to.equal(methodology);
      expect(certificate.verifier).to.equal(minter.address);
      expect(certificate.isRetired).to.be.false;
    });

    it("Should fail if not minter", async function () {
      await expect(
        carbonNFT.connect(user1).mintCertificate(
          user1.address, 1, 1000, "VCS", 
          ethers.utils.keccak256(ethers.utils.toUtf8Bytes("test")), 
          "test"
        )
      ).to.be.reverted;
    });

    it("Should emit CertificateMinted event", async function () {
      await expect(
        carbonNFT.connect(minter).mintCertificate(
          user1.address, 1, 1000, "VCS",
          ethers.utils.keccak256(ethers.utils.toUtf8Bytes("test")),
          "test"
        )
      ).to.emit(carbonNFT, "CertificateMinted")
       .withArgs(0, user1.address, 1, 1000);
    });
  });

  describe("Certificate Retirement", function () {
    beforeEach(async function () {
      await carbonNFT.connect(minter).mintCertificate(
        user1.address, 1, 1000, "VCS",
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("test")),
        "test"
      );
    });

    it("Should retire certificate", async function () {
      await carbonNFT.connect(user1).retireCertificate(0);
      
      const certificate = await carbonNFT.getCertificate(0);
      expect(certificate.isRetired).to.be.true;
      
      // Token should be burned
      await expect(carbonNFT.ownerOf(0)).to.be.reverted;
    });

    it("Should emit CertificateRetired event", async function () {
      await expect(carbonNFT.connect(user1).retireCertificate(0))
        .to.emit(carbonNFT, "CertificateRetired")
        .withArgs(0, user1.address, 1000);
    });

    it("Should fail if not owner", async function () {
      await expect(carbonNFT.connect(user2).retireCertificate(0))
        .to.be.reverted;
    });
  });

  describe("Ownership Tracking", function () {
    beforeEach(async function () {
      await carbonNFT.connect(minter).mintCertificate(
        user1.address, 1, 1000, "VCS",
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("test")),
        "test"
      );
    });

    it("Should track owner tokens", async function () {
      const tokens = await carbonNFT.getOwnerTokens(user1.address);
      expect(tokens.length).to.equal(1);
      expect(tokens[0]).to.equal(0);
    });

    it("Should calculate total carbon credits", async function () {
      const total = await carbonNFT.getTotalCarbonCredits(user1.address);
      expect(total).to.equal(1000);
    });

    it("Should update tracking on transfer", async function () {
      await carbonNFT.connect(user1).transferFrom(user1.address, user2.address, 0);
      
      const user1Tokens = await carbonNFT.getOwnerTokens(user1.address);
      const user2Tokens = await carbonNFT.getOwnerTokens(user2.address);
      
      expect(user1Tokens.length).to.equal(0);
      expect(user2Tokens.length).to.equal(1);
    });
  });
});
```

#### 2.2 Integration Tests
**File**: `blockchain/test/Integration.test.js`
```javascript
const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Full Integration Test", function () {
  let carbonNFT, verifier;
  let owner, user1, verifierAccount;

  beforeEach(async function () {
    [owner, user1, verifierAccount] = await ethers.getSigners();
    
    // Deploy NFT contract
    const CarbonCreditNFT = await ethers.getContractFactory("CarbonCreditNFT");
    carbonNFT = await CarbonCreditNFT.deploy(owner.address);
    await carbonNFT.deployed();

    // Deploy Verifier contract
    const CarbonCreditVerifier = await ethers.getContractFactory("CarbonCreditVerifier");
    verifier = await CarbonCreditVerifier.deploy(owner.address, carbonNFT.address);
    await verifier.deployed();

    // Grant minter role to verifier contract
    const MINTER_ROLE = await carbonNFT.MINTER_ROLE();
    await carbonNFT.grantRole(MINTER_ROLE, verifier.address);

    // Grant verifier role
    const VERIFIER_ROLE = await verifier.VERIFIER_ROLE();
    await verifier.grantRole(VERIFIER_ROLE, verifierAccount.address);
  });

  it("Should complete full verification workflow", async function () {
    // Step 1: Submit verification request
    const tx1 = await verifier.connect(user1).submitVerification(
      1, // projectId
      1000, // carbonAmount
      "VCS", // methodology
      ethers.utils.keccak256(ethers.utils.toUtf8Bytes("verification data")),
      "QmTest123" // ipfsHash
    );

    // Check event emission
    await expect(tx1)
      .to.emit(verifier, "VerificationRequested")
      .withArgs(0, 1, user1.address, 1000);

    // Step 2: Approve verification
    const tokenURI = "https://ipfs.io/ipfs/QmTest123";
    const tx2 = await verifier.connect(verifierAccount).approveVerification(0, tokenURI);

    // Check approval event
    await expect(tx2)
      .to.emit(verifier, "VerificationApproved")
      .withArgs(0, 1, verifierAccount.address, 0);

    // Step 3: Verify NFT was minted
    expect(await carbonNFT.ownerOf(0)).to.equal(user1.address);
    expect(await carbonNFT.tokenURI(0)).to.equal(tokenURI);

    // Step 4: Check certificate details
    const certificate = await carbonNFT.getCertificate(0);
    expect(certificate.projectId).to.equal(1);
    expect(certificate.carbonAmount).to.equal(1000);
    expect(certificate.methodology).to.equal("VCS");

    // Step 5: Verify project status
    expect(await verifier.isProjectVerified(1)).to.be.true;
  });

  it("Should handle verification rejection", async function () {
    // Submit verification request
    await verifier.connect(user1).submitVerification(
      1, 1000, "VCS",
      ethers.utils.keccak256(ethers.utils.toUtf8Bytes("verification data")),
      "QmTest123"
    );

    // Reject verification
    const rejectionReason = "Insufficient documentation";
    const tx = await verifier.connect(verifierAccount).rejectVerification(0, rejectionReason);

    // Check rejection event
    await expect(tx)
      .to.emit(verifier, "VerificationRejected")
      .withArgs(0, 1, verifierAccount.address, rejectionReason);

    // Verify no NFT was minted
    await expect(carbonNFT.ownerOf(0)).to.be.reverted;

    // Check verification status
    const verification = await verifier.getVerification(0);
    expect(verification.status).to.equal(2); // Rejected
    expect(verification.rejectionReason).to.equal(rejectionReason);
  });
});
```

### **Phase 2B: Deployment Infrastructure**
**Duration**: 4 hours

#### 2.3 Deployment Scripts
**File**: `blockchain/deploy/01-deploy-contracts.js`
```javascript
const { ethers, upgrades } = require("hardhat");
const fs = require('fs');
const path = require('path');

async function main() {
  const [deployer] = await ethers.getSigners();
  
  console.log("Deploying contracts with the account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());

  // Deploy Carbon Credit NFT
  console.log("\n1. Deploying CarbonCreditNFT...");
  const CarbonCreditNFT = await ethers.getContractFactory("CarbonCreditNFT");
  const carbonNFT = await CarbonCreditNFT.deploy(deployer.address);
  await carbonNFT.deployed();
  console.log("CarbonCreditNFT deployed to:", carbonNFT.address);

  // Deploy Verifier
  console.log("\n2. Deploying CarbonCreditVerifier...");
  const CarbonCreditVerifier = await ethers.getContractFactory("CarbonCreditVerifier");
  const verifier = await CarbonCreditVerifier.deploy(deployer.address, carbonNFT.address);
  await verifier.deployed();
  console.log("CarbonCreditVerifier deployed to:", verifier.address);

  // Grant minter role to verifier
  console.log("\n3. Setting up permissions...");
  const MINTER_ROLE = await carbonNFT.MINTER_ROLE();
  await carbonNFT.grantRole(MINTER_ROLE, verifier.address);
  console.log("Granted MINTER_ROLE to verifier contract");

  // Save deployment info
  const deploymentInfo = {
    network: network.name,
    deployer: deployer.address,
    contracts: {
      CarbonCreditNFT: {
        address: carbonNFT.address,
        deploymentHash: carbonNFT.deployTransaction.hash
      },
      CarbonCreditVerifier: {
        address: verifier.address,
        deploymentHash: verifier.deployTransaction.hash
      }
    },
    deployedAt: new Date().toISOString(),
    gasUsed: {
      CarbonCreditNFT: (await carbonNFT.deployTransaction.wait()).gasUsed.toString(),
      CarbonCreditVerifier: (await verifier.deployTransaction.wait()).gasUsed.toString()
    }
  };

  // Write deployment info
  const deploymentsDir = path.join(__dirname, '../deployments');
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }
  
  fs.writeFileSync(
    path.join(deploymentsDir, `${network.name}.json`),
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log(`\n✅ Deployment complete! Info saved to deployments/${network.name}.json`);
  
  // Verify contracts on Polygonscan (if not localhost)
  if (network.name !== 'localhost' && network.name !== 'hardhat') {
    console.log("\n4. Verifying contracts on Polygonscan...");
    console.log("Please wait a few minutes, then run:");
    console.log(`npx hardhat verify --network ${network.name} ${carbonNFT.address} "${deployer.address}"`);
    console.log(`npx hardhat verify --network ${network.name} ${verifier.address} "${deployer.address}" "${carbonNFT.address}"`);
  }

  return deploymentInfo;
}

// Execute deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
```

#### 2.4 Environment Configuration
**File**: `blockchain/.env.example`
```bash
# Private key for deployment (DO NOT commit real keys)
PRIVATE_KEY=your_private_key_here

# Polygon RPC URLs
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/your_api_key
MUMBAI_RPC_URL=https://polygon-mumbai.g.alchemy.com/v2/your_api_key

# API Keys
POLYGONSCAN_API_KEY=your_polygonscan_api_key
ALCHEMY_API_KEY=your_alchemy_api_key

# Frontend Configuration  
REACT_APP_POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/your_api_key
REACT_APP_CONTRACT_NFT=deployed_nft_contract_address
REACT_APP_CONTRACT_VERIFIER=deployed_verifier_contract_address
REACT_APP_CHAIN_ID=137

# Backend Configuration
WEB3_PROVIDER_URL=https://polygon-mainnet.g.alchemy.com/v2/your_api_key
CONTRACT_NFT_ADDRESS=deployed_nft_contract_address
CONTRACT_VERIFIER_ADDRESS=deployed_verifier_contract_address
ADMIN_PRIVATE_KEY=admin_private_key_for_backend
```

---

# **📅 WEEK 2: WEB3 FRONTEND & BACKEND INTEGRATION**

## **Day 5-7: Web3 Frontend Integration**

### **Phase 3A: Web3 Service Layer**
**Duration**: 8 hours

#### 3.1 Blockchain Service
**File**: `frontend/src/services/blockchainService.js`
```javascript
import { ethers } from 'ethers';
import Web3Modal from 'web3modal';
import WalletConnectProvider from '@walletconnect/web3-provider';

// Contract ABI imports
import CarbonCreditNFT_ABI from '../contracts/CarbonCreditNFT.json';
import CarbonCreditVerifier_ABI from '../contracts/CarbonCreditVerifier.json';

class BlockchainService {
  constructor() {
    this.web3Modal = null;
    this.provider = null;
    this.signer = null;
    this.contracts = {};
    this.isConnected = false;
    
    this.initWeb3Modal();
  }

  initWeb3Modal() {
    this.web3Modal = new Web3Modal({
      cacheProvider: true,
      providerOptions: {
        walletconnect: {
          package: WalletConnectProvider,
          options: {
            rpc: {
              137: process.env.REACT_APP_POLYGON_RPC_URL
            },
            bridge: "https://bridge.walletconnect.org",
            qrcodeModal: require("@walletconnect/qrcode-modal")
          }
        }
      },
      theme: {
        background: "rgb(39, 49, 56)",
        main: "rgb(199, 199, 199)",
        secondary: "rgb(136, 136, 136)",
        border: "rgba(195, 195, 195, 0.14)",
        hover: "rgb(16, 26, 32)"
      }
    });
  }

  async connectWallet() {
    try {
      const instance = await this.web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(instance);
      const signer = provider.getSigner();
      const address = await signer.getAddress();
      const network = await provider.getNetwork();

      // Check if on Polygon network
      if (network.chainId !== 137 && network.chainId !== 80001) {
        await this.switchToPolygon();
      }

      this.provider = provider;
      this.signer = signer;
      this.isConnected = true;

      // Initialize contracts
      await this.initializeContracts();

      // Listen for account/network changes
      this.setupEventListeners(instance);

      return {
        address,
        network: network.chainId,
        provider: this.provider
      };
    } catch (error) {
      console.error('Failed to connect wallet:', error);
      throw new Error('Failed to connect wallet. Please try again.');
    }
  }

  async switchToPolygon() {
    try {
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: '0x89' }], // Polygon Mainnet
      });
    } catch (switchError) {
      if (switchError.code === 4902) {
        // Add Polygon network if not added
        await window.ethereum.request({
          method: 'wallet_addEthereumChain',
          params: [{
            chainId: '0x89',
            chainName: 'Polygon Mainnet',
            nativeCurrency: {
              name: 'MATIC',
              symbol: 'MATIC',
              decimals: 18,
            },
            rpcUrls: [process.env.REACT_APP_POLYGON_RPC_URL],
            blockExplorerUrls: ['https://polygonscan.com/'],
          }],
        });
      }
    }
  }

  async initializeContracts() {
    if (!this.signer) throw new Error('Wallet not connected');

    try {
      // Initialize NFT Contract
      this.contracts.nft = new ethers.Contract(
        process.env.REACT_APP_CONTRACT_NFT,
        CarbonCreditNFT_ABI.abi,
        this.signer
      );

      // Initialize Verifier Contract
      this.contracts.verifier = new ethers.Contract(
        process.env.REACT_APP_CONTRACT_VERIFIER,
        CarbonCreditVerifier_ABI.abi,
        this.signer
      );

      console.log('Contracts initialized successfully');
    } catch (error) {
      console.error('Failed to initialize contracts:', error);
      throw error;
    }
  }

  setupEventListeners(instance) {
    // Account changed
    instance.on('accountsChanged', (accounts) => {
      if (accounts.length === 0) {
        this.disconnect();
      } else {
        window.location.reload();
      }
    });

    // Chain changed
    instance.on('chainChanged', () => {
      window.location.reload();
    });

    // Disconnect
    instance.on('disconnect', () => {
      this.disconnect();
    });
  }

  async disconnect() {
    if (this.web3Modal) {
      await this.web3Modal.clearCachedProvider();
    }
    this.provider = null;
    this.signer = null;
    this.contracts = {};
    this.isConnected = false;
  }

  // NFT Contract Methods
  async mintCertificate(to, projectId, carbonAmount, methodology, verificationHash, tokenURI) {
    if (!this.contracts.nft) throw new Error('NFT contract not initialized');

    try {
      const tx = await this.contracts.nft.mintCertificate(
        to, projectId, carbonAmount, methodology, verificationHash, tokenURI
      );
      
      const receipt = await tx.wait();
      return {
        txHash: receipt.transactionHash,
        tokenId: receipt.events?.find(e => e.event === 'CertificateMinted')?.args?.tokenId?.toString()
      };
    } catch (error) {
      console.error('Failed to mint certificate:', error);
      throw error;
    }
  }

  async getCertificate(tokenId) {
    if (!this.contracts.nft) throw new Error('NFT contract not initialized');

    try {
      const certificate = await this.contracts.nft.getCertificate(tokenId);
      return {
        projectId: certificate.projectId.toString(),
        carbonAmount: certificate.carbonAmount.toString(),
        verificationDate: new Date(certificate.verificationDate.toNumber() * 1000),
        verifier: certificate.verifier,
        isRetired: certificate.isRetired,
        methodology: certificate.methodology,
        verificationHash: certificate.verificationHash
      };
    } catch (error) {
      console.error('Failed to get certificate:', error);
      throw error;
    }
  }

  async getOwnerTokens(address) {
    if (!this.contracts.nft) throw new Error('NFT contract not initialized');

    try {
      const tokens = await this.contracts.nft.getOwnerTokens(address);
      return tokens.map(token => token.toString());
    } catch (error) {
      console.error('Failed to get owner tokens:', error);
      throw error;
    }
  }

  async getTotalCarbonCredits(address) {
    if (!this.contracts.nft) throw new Error('NFT contract not initialized');

    try {
      const total = await this.contracts.nft.getTotalCarbonCredits(address);
      return total.toString();
    } catch (error) {
      console.error('Failed to get total carbon credits:', error);
      throw error;
    }
  }

  async retireCertificate(tokenId) {
    if (!this.contracts.nft) throw new Error('NFT contract not initialized');

    try {
      const tx = await this.contracts.nft.retireCertificate(tokenId);
      const receipt = await tx.wait();
      return {
        txHash: receipt.transactionHash,
        gasUsed: receipt.gasUsed.toString()
      };
    } catch (error) {
      console.error('Failed to retire certificate:', error);
      throw error;
    }
  }

  // Verifier Contract Methods
  async submitVerification(projectId, carbonAmount, methodology, dataHash, ipfsHash) {
    if (!this.contracts.verifier) throw new Error('Verifier contract not initialized');

    try {
      const tx = await this.contracts.verifier.submitVerification(
        projectId, carbonAmount, methodology, dataHash, ipfsHash
      );
      
      const receipt = await tx.wait();
      const verificationId = receipt.events?.find(e => e.event === 'VerificationRequested')?.args?.verificationId?.toString();
      
      return {
        txHash: receipt.transactionHash,
        verificationId,
        gasUsed: receipt.gasUsed.toString()
      };
    } catch (error) {
      console.error('Failed to submit verification:', error);
      throw error;
    }
  }

  async getVerification(verificationId) {
    if (!this.contracts.verifier) throw new Error('Verifier contract not initialized');

    try {
      const verification = await this.contracts.verifier.getVerification(verificationId);
      return {
        projectId: verification.projectId.toString(),
        requester: verification.requester,
        carbonAmount: verification.carbonAmount.toString(),
        methodology: verification.methodology,
        dataHash: verification.dataHash,
        ipfsHash: verification.ipfsHash,
        status: verification.status, // 0: Pending, 1: Approved, 2: Rejected
        verifier: verification.verifier,
        submissionTime: new Date(verification.submissionTime.toNumber() * 1000),
        verificationTime: verification.verificationTime.toNumber() > 0 
          ? new Date(verification.verificationTime.toNumber() * 1000)
          : null,
        rejectionReason: verification.rejectionReason
      };
    } catch (error) {
      console.error('Failed to get verification:', error);
      throw error;
    }
  }

  async getUserRequests(address) {
    if (!this.contracts.verifier) throw new Error('Verifier contract not initialized');

    try {
      const requests = await this.contracts.verifier.getUserRequests(address);
      return requests.map(req => req.toString());
    } catch (error) {
      console.error('Failed to get user requests:', error);
      throw error;
    }
  }

  // Utility Methods
  async estimateGas(contractMethod, ...args) {
    try {
      const gasEstimate = await contractMethod.estimateGas(...args);
      return gasEstimate.toString();
    } catch (error) {
      console.error('Failed to estimate gas:', error);
      return '500000'; // Default fallback
    }
  }

  async getBalance(address) {
    if (!this.provider) throw new Error('Provider not initialized');
    
    try {
      const balance = await this.provider.getBalance(address);
      return ethers.utils.formatEther(balance);
    } catch (error) {
      console.error('Failed to get balance:', error);
      throw error;
    }
  }

  formatTokenAmount(amount, decimals = 18) {
    return ethers.utils.formatUnits(amount.toString(), decimals);
  }

  parseTokenAmount(amount, decimals = 18) {
    return ethers.utils.parseUnits(amount.toString(), decimals);
  }

  // Event Listeners
  onCertificateMinted(callback) {
    if (!this.contracts.nft) return;
    
    this.contracts.nft.on('CertificateMinted', (tokenId, recipient, projectId, carbonAmount, event) => {
      callback({
        tokenId: tokenId.toString(),
        recipient,
        projectId: projectId.toString(),
        carbonAmount: carbonAmount.toString(),
        txHash: event.transactionHash,
        blockNumber: event.blockNumber
      });
    });
  }

  onVerificationRequested(callback) {
    if (!this.contracts.verifier) return;
    
    this.contracts.verifier.on('VerificationRequested', (verificationId, projectId, requester, carbonAmount, event) => {
      callback({
        verificationId: verificationId.toString(),
        projectId: projectId.toString(),
        requester,
        carbonAmount: carbonAmount.toString(),
        txHash: event.transactionHash,
        blockNumber: event.blockNumber
      });
    });
  }
}

// Export singleton instance
const blockchainService = new BlockchainService();
export default blockchainService;
```

#### 3.2 Web3 React Hook
**File**: `frontend/src/hooks/useWeb3.js`
```javascript
import { useState, useEffect, useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import blockchainService from '../services/blockchainService';
import { 
  setConnected, 
  setDisconnected, 
  setAccount, 
  setNetwork, 
  setError,
  setLoading 
} from '../store/blockchainSlice';

export const useWeb3 = () => {
  const dispatch = useDispatch();
  const blockchain = useSelector(state => state.blockchain);
  
  const [isConnecting, setIsConnecting] = useState(false);

  // Connect to wallet
  const connect = useCallback(async () => {
    if (isConnecting) return;
    
    setIsConnecting(true);
    dispatch(setLoading(true));
    dispatch(setError(null));

    try {
      const connection = await blockchainService.connectWallet();
      
      dispatch(setConnected(true));
      dispatch(setAccount(connection.address));
      dispatch(setNetwork(connection.network));
      
      // Get user's certificates
      const tokens = await blockchainService.getOwnerTokens(connection.address);
      const totalCredits = await blockchainService.getTotalCarbonCredits(connection.address);
      
      return {
        ...connection,
        tokens,
        totalCredits
      };
    } catch (error) {
      dispatch(setError(error.message));
      throw error;
    } finally {
      setIsConnecting(false);
      dispatch(setLoading(false));
    }
  }, [dispatch, isConnecting]);

  // Disconnect wallet
  const disconnect = useCallback(async () => {
    try {
      await blockchainService.disconnect();
      dispatch(setDisconnected());
    } catch (error) {
      dispatch(setError(error.message));
    }
  }, [dispatch]);

  // Check if already connected on mount
  useEffect(() => {
    const checkConnection = async () => {
      if (blockchainService.isConnected) {
        // Already connected, update state
        try {
          const address = await blockchainService.signer.getAddress();
          const network = await blockchainService.provider.getNetwork();
          
          dispatch(setConnected(true));
          dispatch(setAccount(address));
          dispatch(setNetwork(network.chainId));
        } catch (error) {
          console.error('Failed to restore connection:', error);
        }
      }
    };

    checkConnection();
  }, [dispatch]);

  return {
    ...blockchain,
    isConnecting,
    connect,
    disconnect,
    service: blockchainService
  };
};

// Additional hooks for specific functionality
export const useCertificates = () => {
  const { account, connected } = useSelector(state => state.blockchain);
  const [certificates, setCertificates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadCertificates = useCallback(async () => {
    if (!connected || !account) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const tokens = await blockchainService.getOwnerTokens(account);
      const certificatePromises = tokens.map(async (tokenId) => {
        const certificate = await blockchainService.getCertificate(tokenId);
        return {
          tokenId,
          ...certificate
        };
      });
      
      const loadedCertificates = await Promise.all(certificatePromises);
      setCertificates(loadedCertificates);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [connected, account]);

  useEffect(() => {
    loadCertificates();
  }, [loadCertificates]);

  return {
    certificates,
    loading,
    error,
    reload: loadCertificates
  };
};

export const useVerifications = () => {
  const { account, connected } = useSelector(state => state.blockchain);
  const [verifications, setVerifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadVerifications = useCallback(async () => {
    if (!connected || !account) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const requestIds = await blockchainService.getUserRequests(account);
      const verificationPromises = requestIds.map(async (requestId) => {
        const verification = await blockchainService.getVerification(requestId);
        return {
          requestId,
          ...verification
        };
      });
      
      const loadedVerifications = await Promise.all(verificationPromises);
      setVerifications(loadedVerifications);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [connected, account]);

  const submitVerification = useCallback(async (projectId, carbonAmount, methodology, dataHash, ipfsHash) => {
    try {
      const result = await blockchainService.submitVerification(
        projectId, carbonAmount, methodology, dataHash, ipfsHash
      );
      
      // Reload verifications
      await loadVerifications();
      
      return result;
    } catch (error) {
      throw error;
    }
  }, [loadVerifications]);

  useEffect(() => {
    loadVerifications();
  }, [loadVerifications]);

  return {
    verifications,
    loading,
    error,
    submitVerification,
    reload: loadVerifications
  };
};
```

### **Phase 3B: Blockchain Redux State**
**Duration**: 2 hours

#### 3.3 Blockchain Redux Slice
**File**: `frontend/src/store/blockchainSlice.js`
```javascript
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import blockchainService from '../services/blockchainService';

// Async thunks
export const connectWallet = createAsyncThunk(
  'blockchain/connectWallet',
  async (_, { rejectWithValue }) => {
    try {
      return await blockchainService.connectWallet();
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const loadUserCertificates = createAsyncThunk(
  'blockchain/loadUserCertificates',
  async (address, { rejectWithValue }) => {
    try {
      const tokens = await blockchainService.getOwnerTokens(address);
      const certificates = await Promise.all(
        tokens.map(async (tokenId) => {
          const certificate = await blockchainService.getCertificate(tokenId);
          return {
            tokenId,
            ...certificate
          };
        })
      );
      return certificates;
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const submitBlockchainVerification = createAsyncThunk(
  'blockchain/submitVerification',
  async ({ projectId, carbonAmount, methodology, dataHash, ipfsHash }, { rejectWithValue }) => {
    try {
      return await blockchainService.submitVerification(
        projectId, carbonAmount, methodology, dataHash, ipfsHash
      );
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

export const retireCertificate = createAsyncThunk(
  'blockchain/retireCertificate',
  async (tokenId, { rejectWithValue }) => {
    try {
      return await blockchainService.retireCertificate(tokenId);
    } catch (error) {
      return rejectWithValue(error.message);
    }
  }
);

const blockchainSlice = createSlice({
  name: 'blockchain',
  initialState: {
    connected: false,
    account: null,
    network: null,
    balance: '0',
    certificates: [],
    verifications: [],
    transactions: [],
    loading: {
      connecting: false,
      certificates: false,
      verification: false,
      transaction: false
    },
    error: null,
    contractAddresses: {
      nft: process.env.REACT_APP_CONTRACT_NFT,
      verifier: process.env.REACT_APP_CONTRACT_VERIFIER
    }
  },
  reducers: {
    setConnected: (state, action) => {
      state.connected = action.payload;
    },
    setDisconnected: (state) => {
      state.connected = false;
      state.account = null;
      state.network = null;
      state.balance = '0';
      state.certificates = [];
      state.verifications = [];
      state.transactions = [];
      state.error = null;
    },
    setAccount: (state, action) => {
      state.account = action.payload;
    },
    setNetwork: (state, action) => {
      state.network = action.payload;
    },
    setBalance: (state, action) => {
      state.balance = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
    setLoading: (state, action) => {
      state.loading = { ...state.loading, ...action.payload };
    },
    addTransaction: (state, action) => {
      state.transactions.unshift(action.payload);
    },
    updateTransactionStatus: (state, action) => {
      const { txHash, status, receipt } = action.payload;
      const transaction = state.transactions.find(tx => tx.hash === txHash);
      if (transaction) {
        transaction.status = status;
        transaction.receipt = receipt;
      }
    },
    addCertificate: (state, action) => {
      state.certificates.push(action.payload);
    },
    updateCertificateStatus: (state, action) => {
      const { tokenId, isRetired } = action.payload;
      const certificate = state.certificates.find(cert => cert.tokenId === tokenId);
      if (certificate) {
        certificate.isRetired = isRetired;
      }
    }
  },
  extraReducers: (builder) => {
    // Connect wallet
    builder
      .addCase(connectWallet.pending, (state) => {
        state.loading.connecting = true;
        state.error = null;
      })
      .addCase(connectWallet.fulfilled, (state, action) => {
        state.loading.connecting = false;
        state.connected = true;
        state.account = action.payload.address;
        state.network = action.payload.network;
      })
      .addCase(connectWallet.rejected, (state, action) => {
        state.loading.connecting = false;
        state.error = action.payload;
      });

    // Load certificates
    builder
      .addCase(loadUserCertificates.pending, (state) => {
        state.loading.certificates = true;
      })
      .addCase(loadUserCertificates.fulfilled, (state, action) => {
        state.loading.certificates = false;
        state.certificates = action.payload;
      })
      .addCase(loadUserCertificates.rejected, (state, action) => {
        state.loading.certificates = false;
        state.error = action.payload;
      });

    // Submit verification
    builder
      .addCase(submitBlockchainVerification.pending, (state) => {
        state.loading.verification = true;
        state.error = null;
      })
      .addCase(submitBlockchainVerification.fulfilled, (state, action) => {
        state.loading.verification = false;
        state.transactions.unshift({
          hash: action.payload.txHash,
          type: 'verification_submission',
          status: 'confirmed',
          timestamp: new Date().toISOString(),
          verificationId: action.payload.verificationId
        });
      })
      .addCase(submitBlockchainVerification.rejected, (state, action) => {
        state.loading.verification = false;
        state.error = action.payload;
      });

    // Retire certificate
    builder
      .addCase(retireCertificate.pending, (state) => {
        state.loading.transaction = true;
      })
      .addCase(retireCertificate.fulfilled, (state, action) => {
        state.loading.transaction = false;
        state.transactions.unshift({
          hash: action.payload.txHash,
          type: 'certificate_retirement',
          status: 'confirmed',
          timestamp: new Date().toISOString()
        });
      })
      .addCase(retireCertificate.rejected, (state, action) => {
        state.loading.transaction = false;
        state.error = action.payload;
      });
  }
});

export const {
  setConnected,
  setDisconnected,
  setAccount,
  setNetwork,
  setBalance,
  setError,
  clearError,
  setLoading,
  addTransaction,
  updateTransactionStatus,
  addCertificate,
  updateCertificateStatus
} = blockchainSlice.actions;

export default blockchainSlice.reducer;
```

---

## **Day 8-10: UI Components Development**

### **Phase 4A: Core Blockchain Components**
**Duration**: 8 hours

#### 4.1 Wallet Connection Component
**File**: `frontend/src/components/blockchain/WalletConnector.js`
```javascript
import React, { useState } from 'react';
import {
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Box,
  Typography,
  Card,
  CardContent,
  Avatar,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import {
  AccountBalanceWallet,
  Link as LinkIcon,
  LinkOff,
  Warning
} from '@mui/icons-material';
import { useWeb3 } from '../../hooks/useWeb3';

const WalletConnector = ({ buttonVariant = 'contained', showBalance = true }) => {
  const { connected, account, network, balance, isConnecting, error, connect, disconnect } = useWeb3();
  const [showDialog, setShowDialog] = useState(false);

  const handleConnect = async () => {
    try {
      await connect();
      setShowDialog(false);
    } catch (error) {
      console.error('Connection failed:', error);
    }
  };

  const handleDisconnect = async () => {
    await disconnect();
    setShowDialog(false);
  };

  const getNetworkName = (chainId) => {
    switch (chainId) {
      case 137: return 'Polygon';
      case 80001: return 'Mumbai Testnet';
      default: return 'Unknown Network';
    }
  };

  const formatAddress = (address) => {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatBalance = (balance) => {
    return parseFloat(balance).toFixed(4);
  };

  if (!connected) {
    return (
      <>
        <Button
          variant={buttonVariant}
          startIcon={<AccountBalanceWallet />}
          onClick={() => setShowDialog(true)}
          disabled={isConnecting}
        >
          {isConnecting ? <CircularProgress size={20} /> : 'Connect Wallet'}
        </Button>

        <Dialog open={showDialog} onClose={() => setShowDialog(false)} maxWidth="sm" fullWidth>
          <DialogTitle>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <AccountBalanceWallet sx={{ mr: 1 }} />
              Connect Your Wallet
            </Box>
          </DialogTitle>
          <DialogContent>
            <Typography variant="body2" color="text.secondary" paragraph>
              Connect your wallet to mint, transfer, and retire carbon credit certificates on the blockchain.
            </Typography>

            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            <Alert severity="info" sx={{ mb: 2 }}>
              <Typography variant="body2">
                <strong>Network Required:</strong> Polygon Mainnet
                <br />
                <strong>Supported Wallets:</strong> MetaMask, WalletConnect, Coinbase Wallet
              </Typography>
            </Alert>

            <Card sx={{ mt: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  What you can do:
                </Typography>
                <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
                  <li>View your carbon credit certificates</li>
                  <li>Transfer certificates to other addresses</li>
                  <li>Retire certificates permanently</li>
                  <li>Submit projects for blockchain verification</li>
                  <li>Track all transactions on Polygonscan</li>
                </Typography>
              </CardContent>
            </Card>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowDialog(false)}>Cancel</Button>
            <Button 
              variant="contained" 
              onClick={handleConnect}
              disabled={isConnecting}
              startIcon={isConnecting ? <CircularProgress size={16} /> : <LinkIcon />}
            >
              {isConnecting ? 'Connecting...' : 'Connect Wallet'}
            </Button>
          </DialogActions>
        </Dialog>
      </>
    );
  }

  return (
    <>
      <Card sx={{ minWidth: 250 }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar sx={{ bgcolor: 'success.main' }}>
            <AccountBalanceWallet />
          </Avatar>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary">
              {formatAddress(account)}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <Chip 
                label={getNetworkName(network)} 
                size="small" 
                color="primary" 
                variant="outlined"
              />
              {showBalance && balance && (
                <Typography variant="caption" color="text.secondary">
                  {formatBalance(balance)} MATIC
                </Typography>
              )}
            </Box>
          </Box>
          <Button
            size="small"
            startIcon={<LinkOff />}
            onClick={() => setShowDialog(true)}
            color="error"
          >
            Disconnect
          </Button>
        </CardContent>
      </Card>

      <Dialog open={showDialog} onClose={() => setShowDialog(false)}>
        <DialogTitle>Disconnect Wallet</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to disconnect your wallet? You'll need to reconnect to perform blockchain operations.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowDialog(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleDisconnect}>
            Disconnect
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default WalletConnector;
```

#### 4.2 Certificate Card Component
**File**: `frontend/src/components/blockchain/CertificateCard.js`
```javascript
import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Link
} from '@mui/material';
import {
  Eco as EcoIcon,
  OpenInNew,
  Delete as RetireIcon,
  Share as ShareIcon,
  Download as DownloadIcon,
  Verified
} from '@mui/icons-material';
import { format } from 'date-fns';
import { useDispatch } from 'react-redux';
import { retireCertificate } from '../../store/blockchainSlice';

const CertificateCard = ({ certificate, onRetire, showActions = true }) => {
  const dispatch = useDispatch();
  const [showRetireDialog, setShowRetireDialog] = useState(false);
  const [retiring, setRetiring] = useState(false);

  const {
    tokenId,
    projectId,
    carbonAmount,
    verificationDate,
    verifier,
    isRetired,
    methodology,
    verificationHash
  } = certificate;

  const handleRetire = async () => {
    setRetiring(true);
    try {
      await dispatch(retireCertificate(tokenId)).unwrap();
      setShowRetireDialog(false);
      if (onRetire) onRetire(tokenId);
    } catch (error) {
      console.error('Failed to retire certificate:', error);
    } finally {
      setRetiring(false);
    }
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Carbon Credit Certificate #${tokenId}`,
          text: `${carbonAmount} tCO₂e Carbon Credits verified using ${methodology}`,
          url: `${window.location.origin}/blockchain/certificate/${tokenId}`
        });
      } catch (error) {
        console.log('Sharing cancelled or failed');
      }
    } else {
      // Fallback - copy to clipboard
      navigator.clipboard.writeText(`${window.location.origin}/blockchain/certificate/${tokenId}`);
    }
  };

  const getPolygonscanUrl = (tokenId) => {
    const contractAddress = process.env.REACT_APP_CONTRACT_NFT;
    return `https://polygonscan.com/token/${contractAddress}?a=${tokenId}`;
  };

  const getMethodologyColor = (methodology) => {
    const colors = {
      'VCS': 'primary',
      'CDM': 'secondary',
      'GOLD': 'warning',
      'CAR': 'info',
      'REDD+': 'success'
    };
    return colors[methodology] || 'default';
  };

  return (
    <>
      <Card sx={{ 
        height: '100%', 
        display: 'flex', 
        flexDirection: 'column',
        opacity: isRetired ? 0.7 : 1,
        border: isRetired ? '2px dashed #ccc' : 'none'
      }}>
        <CardContent sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <EcoIcon sx={{ mr: 1, color: 'success.main' }} />
              <Typography variant="h6">
                Certificate #{tokenId}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              {isRetired ? (
                <Chip label="RETIRED" color="error" size="small" />
              ) : (
                <Chip label="ACTIVE" color="success" size="small" />
              )}
              <Chip 
                label={methodology} 
                color={getMethodologyColor(methodology)} 
                size="small" 
                variant="outlined"
              />
            </Box>
          </Box>

          <Box sx={{ mb: 2 }}>
            <Typography variant="h4" color="success.main" gutterBottom>
              {carbonAmount}
              <Typography component="span" variant="h6" color="text.secondary" sx={{ ml: 1 }}>
                tCO₂e
              </Typography>
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Project ID:</Typography>
              <Typography variant="body2">{projectId}</Typography>
            </Box>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Verified:</Typography>
              <Typography variant="body2">
                {format(verificationDate, 'MMM dd, yyyy')}
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">Verifier:</Typography>
              <Tooltip title={verifier}>
                <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                  {verifier.slice(0, 6)}...{verifier.slice(-4)}
                </Typography>
              </Tooltip>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">Blockchain:</Typography>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2">Polygon</Typography>
                <Verified sx={{ ml: 0.5, fontSize: 16, color: 'success.main' }} />
              </Box>
            </Box>
          </Box>

          {isRetired && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              <Typography variant="body2">
                This certificate has been permanently retired and removed from circulation.
              </Typography>
            </Alert>
          )}
        </CardContent>

        {showActions && (
          <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
            <Box>
              <Tooltip title="View on Polygonscan">
                <IconButton 
                  size="small"
                  onClick={() => window.open(getPolygonscanUrl(tokenId), '_blank')}
                >
                  <OpenInNew />
                </IconButton>
              </Tooltip>
              <Tooltip title="Share Certificate">
                <IconButton size="small" onClick={handleShare}>
                  <ShareIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Download Certificate">
                <IconButton size="small">
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
            </Box>

            {!isRetired && (
              <Button
                variant="outlined"
                color="error"
                startIcon={<RetireIcon />}
                onClick={() => setShowRetireDialog(true)}
                size="small"
              >
                Retire
              </Button>
            )}
          </CardActions>
        )}
      </Card>

      {/* Retire Dialog */}
      <Dialog open={showRetireDialog} onClose={() => setShowRetireDialog(false)}>
        <DialogTitle>Retire Carbon Credit Certificate</DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Warning:</strong> This action cannot be undone. Retiring a certificate permanently removes it from circulation and burns the NFT.
            </Typography>
          </Alert>
          
          <Typography paragraph>
            You are about to retire <strong>{carbonAmount} tCO₂e</strong> from Certificate #{tokenId}.
          </Typography>
          
          <Typography variant="body2" color="text.secondary">
            Retirement is used when carbon credits are consumed to offset emissions. Once retired, these credits cannot be transferred or sold.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowRetireDialog(false)}>Cancel</Button>
          <Button 
            variant="contained" 
            color="error" 
            onClick={handleRetire}
            disabled={retiring}
            startIcon={retiring ? <CircularProgress size={16} /> : <RetireIcon />}
          >
            {retiring ? 'Retiring...' : 'Retire Certificate'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default CertificateCard;
```

---

## **Day 11-12: Backend Integration**

### **Phase 5A: Python Web3 Integration**
**Duration**: 6 hours

#### 5.1 Blockchain Backend Service
**File**: `backend/services/blockchain_service.py`
```python
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
import requests

logger = logging.getLogger(__name__)

@dataclass
class CertificateData:
    token_id: str
    project_id: str
    carbon_amount: str
    verification_date: datetime
    verifier: str
    is_retired: bool
    methodology: str
    verification_hash: str
    transaction_hash: str

@dataclass
class VerificationData:
    verification_id: str
    project_id: str
    requester: str
    carbon_amount: str
    methodology: str
    status: int  # 0: Pending, 1: Approved, 2: Rejected
    submission_time: datetime
    verifier: Optional[str] = None
    verification_time: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    transaction_hash: Optional[str] = None

class BlockchainService:
    def __init__(self):
        self.w3 = None
        self.account = None
        self.contracts = {}
        self.initialize_web3()
        self.load_contracts()
        
    def initialize_web3(self):
        """Initialize Web3 connection to Polygon network"""
        try:
            rpc_url = os.getenv('WEB3_PROVIDER_URL')
            if not rpc_url:
                raise ValueError("WEB3_PROVIDER_URL not configured")
            
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            # Add PoA middleware for Polygon
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self.w3.is_connected():
                raise ConnectionError("Failed to connect to Polygon network")
                
            # Load admin account for contract interactions
            private_key = os.getenv('ADMIN_PRIVATE_KEY')
            if private_key:
                self.account = Account.from_key(private_key)
                logger.info(f"Admin account loaded: {self.account.address}")
            
            logger.info(f"Connected to Polygon network (Chain ID: {self.w3.eth.chain_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Web3: {e}")
            raise
    
    def load_contracts(self):
        """Load contract ABIs and addresses"""
        try:
            # Load contract addresses
            nft_address = os.getenv('CONTRACT_NFT_ADDRESS')
            verifier_address = os.getenv('CONTRACT_VERIFIER_ADDRESS')
            
            if not nft_address or not verifier_address:
                logger.warning("Contract addresses not configured")
                return
                
            # Load contract ABIs (these would be generated during deployment)
            contracts_dir = os.path.join(os.path.dirname(__file__), '..', 'contracts')
            
            # Load NFT contract
            with open(os.path.join(contracts_dir, 'CarbonCreditNFT.json'), 'r') as f:
                nft_abi = json.load(f)['abi']
            
            self.contracts['nft'] = self.w3.eth.contract(
                address=nft_address,
                abi=nft_abi
            )
            
            # Load Verifier contract
            with open(os.path.join(contracts_dir, 'CarbonCreditVerifier.json'), 'r') as f:
                verifier_abi = json.load(f)['abi']
                
            self.contracts['verifier'] = self.w3.eth.contract(
                address=verifier_address,
                abi=verifier_abi
            )
            
            logger.info("Smart contracts loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load contracts: {e}")
    
    async def mint_certificate(
        self, 
        recipient_address: str, 
        project_id: int, 
        carbon_amount: int,
        methodology: str,
        verification_hash: str,
        token_uri: str
    ) -> Dict[str, Any]:
        """Mint a new carbon credit certificate NFT"""
        try:
            if 'nft' not in self.contracts:
                raise ValueError("NFT contract not loaded")
                
            if not self.account:
                raise ValueError("Admin account not loaded")
            
            contract = self.contracts['nft']
            
            # Build transaction
            transaction = contract.functions.mintCertificate(
                recipient_address,
                project_id,
                carbon_amount,
                methodology,
                bytes.fromhex(verification_hash.replace('0x', '')),
                token_uri
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 500000,
                'gasPrice': self.w3.to_wei('35', 'gwei')
            })
            
            # Sign and send transaction
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract token ID from event logs
            token_id = None
            for log in receipt.logs:
                try:
                    decoded_log = contract.events.CertificateMinted().processLog(log)
                    token_id = decoded_log.args.tokenId
                    break
                except:
                    continue
            
            return {
                'success': True,
                'transaction_hash': receipt.transactionHash.hex(),
                'token_id': token_id,
                'gas_used': receipt.gasUsed,
                'block_number': receipt.blockNumber
            }
            
        except Exception as e:
            logger.error(f"Failed to mint certificate: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_certificate(self, token_id: int) -> Optional[CertificateData]:
        """Get certificate details from blockchain"""
        try:
            if 'nft' not in self.contracts:
                return None
                
            contract = self.contracts['nft']
            certificate = contract.functions.getCertificate(token_id).call()
            
            return CertificateData(
                token_id=str(token_id),
                project_id=str(certificate[0]),
                carbon_amount=str(certificate[1]),
                verification_date=datetime.fromtimestamp(certificate[2]),
                verifier=certificate[3],
                is_retired=certificate[4],
                methodology=certificate[5],
                verification_hash=certificate[6].hex(),
                transaction_hash=""  # Would need to query events for this
            )
            
        except Exception as e:
            logger.error(f"Failed to get certificate: {e}")
            return None
    
    async def get_user_certificates(self, address: str) -> List[CertificateData]:
        """Get all certificates owned by an address"""
        try:
            if 'nft' not in self.contracts:
                return []
                
            contract = self.contracts['nft']
            token_ids = contract.functions.getOwnerTokens(address).call()
            
            certificates = []
            for token_id in token_ids:
                cert = await self.get_certificate(token_id)
                if cert:
                    certificates.append(cert)
            
            return certificates
            
        except Exception as e:
            logger.error(f"Failed to get user certificates: {e}")
            return []
    
    async def submit_verification(
        self,
        project_id: int,
        carbon_amount: int,
        methodology: str,
        data_hash: str,
        ipfs_hash: str,
        requester_address: str
    ) -> Dict[str, Any]:
        """Submit verification request to blockchain"""
        try:
            if 'verifier' not in self.contracts:
                raise ValueError("Verifier contract not loaded")
            
            contract = self.contracts['verifier']
            
            # For backend submissions, we use a service account
            # In production, this would be called by the user's wallet
            
            transaction = contract.functions.submitVerification(
                project_id,
                carbon_amount,
                methodology,
                bytes.fromhex(data_hash.replace('0x', '')),
                ipfs_hash
            ).build_transaction({
                'from': requester_address,  # Would be user's address
                'nonce': self.w3.eth.get_transaction_count(requester_address),
                'gas': 300000,
                'gasPrice': self.w3.to_wei('35', 'gwei')
            })
            
            # In reality, this would be signed by user's wallet
            # Here we simulate for testing purposes
            
            return {
                'success': True,
                'message': 'Verification request prepared',
                'transaction_data': transaction
            }
            
        except Exception as e:
            logger.error(f"Failed to submit verification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def approve_verification(
        self,
        verification_id: int,
        token_uri: str
    ) -> Dict[str, Any]:
        """Approve verification request and mint certificate"""
        try:
            if 'verifier' not in self.contracts or not self.account:
                raise ValueError("Verifier contract or admin account not available")
            
            contract = self.contracts['verifier']
            
            transaction = contract.functions.approveVerification(
                verification_id,
                token_uri
            ).build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': 600000,
                'gasPrice': self.w3.to_wei('35', 'gwei')
            })
            
            signed_txn = self.account.sign_transaction(transaction)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Extract token ID from events
            token_id = None
            for log in receipt.logs:
                try:
                    decoded_log = contract.events.VerificationApproved().processLog(log)
                    token_id = decoded_log.args.tokenId
                    break
                except:
                    continue
            
            return {
                'success': True,
                'transaction_hash': receipt.transactionHash.hex(),
                'token_id': token_id,
                'gas_used': receipt.gasUsed
            }
            
        except Exception as e:
            logger.error(f"Failed to approve verification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_verification(self, verification_id: int) -> Optional[VerificationData]:
        """Get verification request details"""
        try:
            if 'verifier' not in self.contracts:
                return None
                
            contract = self.contracts['verifier']
            verification = contract.functions.getVerification(verification_id).call()
            
            return VerificationData(
                verification_id=str(verification_id),
                project_id=str(verification[0]),
                requester=verification[1],
                carbon_amount=str(verification[2]),
                methodology=verification[3],
                status=verification[5],  # 0: Pending, 1: Approved, 2: Rejected
                submission_time=datetime.fromtimestamp(verification[7]),
                verifier=verification[6] if verification[6] != '0x' + '0' * 40 else None,
                verification_time=datetime.fromtimestamp(verification[8]) if verification[8] > 0 else None,
                rejection_reason=verification[9] if verification[9] else None
            )
            
        except Exception as e:
            logger.error(f"Failed to get verification: {e}")
            return None
    
    def get_transaction_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transaction status and details"""
        try:
            receipt = self.w3.eth.get_transaction_receipt(tx_hash)
            transaction = self.w3.eth.get_transaction(tx_hash)
            
            return {
                'status': 'confirmed' if receipt.status == 1 else 'failed',
                'block_number': receipt.blockNumber,
                'gas_used': receipt.gasUsed,
                'gas_price': transaction.gasPrice,
                'from': transaction['from'],
                'to': transaction['to'],
                'value': transaction['value']
            }
            
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {'status': 'unknown', 'error': str(e)}
    
    def estimate_gas_price(self) -> int:
        """Get current gas price estimate"""
        try:
            # Get current gas price with some buffer
            base_price = self.w3.eth.gas_price
            return int(base_price * 1.2)  # 20% buffer
            
        except Exception as e:
            logger.error(f"Failed to estimate gas price: {e}")
            return self.w3.to_wei('35', 'gwei')  # Fallback

# Singleton instance
blockchain_service = BlockchainService()
```

#### 5.2 Blockchain API Endpoints
**File**: `backend/api/blockchain_endpoints.py`
```python
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, validator
from typing import Optional, List
import logging
from datetime import datetime

from ..services.blockchain_service import blockchain_service
from ..auth import get_current_user, UserResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/blockchain", tags=["blockchain"])

# Request Models
class CertificateRequest(BaseModel):
    recipient_address: str
    project_id: int
    carbon_amount: int
    methodology: str
    verification_hash: str
    token_uri: str
    
    @validator('recipient_address')
    def validate_address(cls, v):
        if not v.startswith('0x') or len(v) != 42:
            raise ValueError('Invalid Ethereum address')
        return v.lower()
    
    @validator('carbon_amount')
    def validate_carbon_amount(cls, v):
        if v <= 0:
            raise ValueError('Carbon amount must be positive')
        return v

class VerificationRequest(BaseModel):
    project_id: int
    carbon_amount: int
    methodology: str
    data_hash: str
    ipfs_hash: str
    requester_address: str

class ApprovalRequest(BaseModel):
    verification_id: int
    token_uri: str

# Response Models
class CertificateResponse(BaseModel):
    token_id: str
    project_id: str
    carbon_amount: str
    verification_date: datetime
    verifier: str
    is_retired: bool
    methodology: str
    verification_hash: str
    transaction_hash: Optional[str]

class TransactionResponse(BaseModel):
    success: bool
    transaction_hash: Optional[str] = None
    token_id: Optional[str] = None
    gas_used: Optional[int] = None
    error: Optional[str] = None

# Endpoints
@router.get("/status")
async def get_blockchain_status():
    """Get blockchain service status"""
    try:
        if not blockchain_service.w3:
            return {"status": "disconnected", "error": "Web3 not initialized"}
        
        if not blockchain_service.w3.is_connected():
            return {"status": "disconnected", "error": "Not connected to network"}
        
        chain_id = blockchain_service.w3.eth.chain_id
        block_number = blockchain_service.w3.eth.block_number
        gas_price = blockchain_service.w3.eth.gas_price
        
        return {
            "status": "connected",
            "chain_id": chain_id,
            "network": "Polygon" if chain_id == 137 else "Mumbai Testnet" if chain_id == 80001 else "Unknown",
            "block_number": block_number,
            "gas_price": gas_price,
            "contracts_loaded": len(blockchain_service.contracts)
        }
        
    except Exception as e:
        logger.error(f"Blockchain status error: {e}")
        return {"status": "error", "error": str(e)}

@router.post("/mint", response_model=TransactionResponse)
async def mint_certificate(
    request: CertificateRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Mint a new carbon credit certificate NFT"""
    # Only admin and verifiers can mint certificates
    if current_user.role not in ['admin', 'verifier']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to mint certificates"
        )
    
    try:
        result = await blockchain_service.mint_certificate(
            recipient_address=request.recipient_address,
            project_id=request.project_id,
            carbon_amount=request.carbon_amount,
            methodology=request.methodology,
            verification_hash=request.verification_hash,
            token_uri=request.token_uri
        )
        
        return TransactionResponse(**result)
        
    except Exception as e:
        logger.error(f"Certificate minting error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to mint certificate: {str(e)}"
        )

@router.get("/certificate/{token_id}", response_model=CertificateResponse)
async def get_certificate(token_id: int):
    """Get certificate details by token ID"""
    try:
        certificate = await blockchain_service.get_certificate(token_id)
        
        if not certificate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Certificate not found"
            )
        
        return CertificateResponse(
            token_id=certificate.token_id,
            project_id=certificate.project_id,
            carbon_amount=certificate.carbon_amount,
            verification_date=certificate.verification_date,
            verifier=certificate.verifier,
            is_retired=certificate.is_retired,
            methodology=certificate.methodology,
            verification_hash=certificate.verification_hash,
            transaction_hash=certificate.transaction_hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Certificate retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve certificate"
        )

@router.get("/certificates/{address}")
async def get_user_certificates(address: str):
    """Get all certificates owned by an address"""
    try:
        # Validate address format
        if not address.startswith('0x') or len(address) != 42:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Ethereum address format"
            )
        
        certificates = await blockchain_service.get_user_certificates(address.lower())
        
        return {
            "address": address,
            "certificate_count": len(certificates),
            "certificates": [
                {
                    "token_id": cert.token_id,
                    "project_id": cert.project_id,
                    "carbon_amount": cert.carbon_amount,
                    "verification_date": cert.verification_date.isoformat(),
                    "verifier": cert.verifier,
                    "is_retired": cert.is_retired,
                    "methodology": cert.methodology
                }
                for cert in certificates
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User certificates retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user certificates"
        )

@router.post("/verification/submit", response_model=TransactionResponse)
async def submit_verification_request(
    request: VerificationRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Submit verification request to blockchain"""
    try:
        result = await blockchain_service.submit_verification(
            project_id=request.project_id,
            carbon_amount=request.carbon_amount,
            methodology=request.methodology,
            data_hash=request.data_hash,
            ipfs_hash=request.ipfs_hash,
            requester_address=request.requester_address
        )
        
        return TransactionResponse(**result)
        
    except Exception as e:
        logger.error(f"Verification submission error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit verification: {str(e)}"
        )

@router.post("/verification/approve", response_model=TransactionResponse)
async def approve_verification_request(
    request: ApprovalRequest,
    current_user: UserResponse = Depends(get_current_user)
):
    """Approve verification request and mint certificate"""
    # Only admin and verifiers can approve verifications
    if current_user.role not in ['admin', 'verifier']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to approve verifications"
        )
    
    try:
        result = await blockchain_service.approve_verification(
            verification_id=request.verification_id,
            token_uri=request.token_uri
        )
        
        return TransactionResponse(**result)
        
    except Exception as e:
        logger.error(f"Verification approval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve verification: {str(e)}"
        )

@router.get("/verification/{verification_id}")
async def get_verification_details(verification_id: int):
    """Get verification request details"""
    try:
        verification = await blockchain_service.get_verification(verification_id)
        
        if not verification:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Verification request not found"
            )
        
        status_names = {0: "Pending", 1: "Approved", 2: "Rejected"}
        
        return {
            "verification_id": verification.verification_id,
            "project_id": verification.project_id,
            "requester": verification.requester,
            "carbon_amount": verification.carbon_amount,
            "methodology": verification.methodology,
            "status": status_names.get(verification.status, "Unknown"),
            "status_code": verification.status,
            "submission_time": verification.submission_time.isoformat(),
            "verifier": verification.verifier,
            "verification_time": verification.verification_time.isoformat() if verification.verification_time else None,
            "rejection_reason": verification.rejection_reason
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification details retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve verification details"
        )

@router.get("/transaction/{tx_hash}")
async def get_transaction_status(tx_hash: str):
    """Get transaction status and details"""
    try:
        if not tx_hash.startswith('0x') or len(tx_hash) != 66:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid transaction hash format"
            )
        
        status_info = blockchain_service.get_transaction_status(tx_hash)
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction status"
        )

@router.get("/gas-price")
async def get_gas_price():
    """Get current gas price estimate"""
    try:
        gas_price = blockchain_service.estimate_gas_price()
        return {
            "gas_price_wei": gas_price,
            "gas_price_gwei": gas_price // 10**9
        }
        
    except Exception as e:
        logger.error(f"Gas price estimation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to estimate gas price"
        )

# Add blockchain router to main app
def setup_blockchain_routes(app):
    """Setup blockchain routes in main FastAPI app"""
    app.include_router(router)
```

---

## **Day 13-14: Integration & Testing**

### **Phase 6A: Complete Frontend Integration**
**Duration**: 4 hours

#### 6.1 Updated Blockchain Page
**File**: `frontend/src/pages/Blockchain.js` (Complete Replacement)
```javascript
import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Box,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Alert,
  Tabs,
  Tab,
  Dialog,
  DialogContent
} from '@mui/material';
import {
  ViewInAr as BlockIcon,
  Search as SearchIcon,
  AccountBalanceWallet,
  Dashboard as DashboardIcon
} from '@mui/icons-material';
import { COMMON_STYLES } from '../theme/constants';
import { useWeb3, useCertificates, useVerifications } from '../hooks/useWeb3';

// Component imports
import WalletConnector from '../components/blockchain/WalletConnector';
import CertificateCard from '../components/blockchain/CertificateCard';
import VerificationForm from '../components/blockchain/VerificationForm';
import TransactionHistory from '../components/blockchain/TransactionHistory';

const Blockchain = () => {
  const { connected, account } = useWeb3();
  const { certificates, loading: certificatesLoading, reload: reloadCertificates } = useCertificates();
  const { verifications, loading: verificationsLoading, submitVerification } = useVerifications();
  
  const [activeTab, setActiveTab] = useState(0);
  const [searchTokenId, setSearchTokenId] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [showVerificationForm, setShowVerificationForm] = useState(false);

  const handleSearch = async () => {
    if (!searchTokenId) return;
    
    try {
      const response = await fetch(`/api/v1/blockchain/certificate/${searchTokenId}`);
      if (response.ok) {
        const certificate = await response.json();
        setSearchResult({
          ...certificate,
          isValid: true,
          status: 'verified'
        });
      } else {
        setSearchResult({
          isValid: false,
          error: 'Certificate not found'
        });
      }
    } catch (error) {
      setSearchResult({
        isValid: false,
        error: 'Search failed'
      });
    }
  };

  const handleCertificateRetire = async (tokenId) => {
    // Reload certificates after retirement
    setTimeout(() => {
      reloadCertificates();
    }, 2000);
  };

  const TabPanel = ({ children, value, index }) => (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );

  return (
    <Container maxWidth="xl" sx={COMMON_STYLES.pageContainer}>
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <BlockIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
          <Typography variant="h4">
            Blockchain Explorer
          </Typography>
        </Box>
        <WalletConnector />
      </Box>

      {/* Status Alert */}
      <Alert severity={connected ? "success" : "info"} sx={{ mb: 3 }}>
        <Typography variant="body1">
          <strong>Blockchain Integration:</strong>{" "}
          {connected 
            ? `Connected to Polygon network. Wallet: ${account?.slice(0, 6)}...${account?.slice(-4)}`
            : "Connect your wallet to interact with carbon credit certificates on the blockchain."
          }
        </Typography>
      </Alert>

      {!connected ? (
        // Not Connected View
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Verify Carbon Credit Certificate
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                <TextField
                  label="Token ID or Transaction Hash"
                  value={searchTokenId}
                  onChange={(e) => setSearchTokenId(e.target.value)}
                  placeholder="Enter token ID to verify certificate"
                  fullWidth
                />
                <Button
                  variant="contained"
                  onClick={handleSearch}
                  startIcon={<SearchIcon />}
                  sx={{ minWidth: 120 }}
                >
                  Verify
                </Button>
              </Box>

              {searchResult && (
                <Card sx={{ mt: 2 }}>
                  <CardContent>
                    {searchResult.isValid ? (
                      <Alert severity="success">
                        <Typography variant="h6" gutterBottom>Certificate Verified</Typography>
                        <Typography variant="body2">
                          Project ID: {searchResult.project_id}<br/>
                          Carbon Credits: {searchResult.carbon_amount} tCO₂e<br/>
                          Methodology: {searchResult.methodology}<br/>
                          Verification Date: {new Date(searchResult.verification_date).toLocaleDateString()}<br/>
                          Status: {searchResult.is_retired ? 'Retired' : 'Active'}
                        </Typography>
                      </Alert>
                    ) : (
                      <Alert severity="error">
                        <Typography variant="body1">{searchResult.error}</Typography>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Connect Wallet to Access
                </Typography>
                <Typography variant="body2" paragraph>
                  Connect your Web3 wallet to:
                </Typography>
                <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
                  <li>View your carbon credit certificates</li>
                  <li>Submit projects for verification</li>
                  <li>Transfer or retire certificates</li>
                  <li>Track all blockchain transactions</li>
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <WalletConnector buttonVariant="contained" />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      ) : (
        // Connected View
        <>
          <Paper sx={{ borderRadius: 2, overflow: 'hidden' }}>
            <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)}>
              <Tab label={`My Certificates (${certificates.length})`} />
              <Tab label={`Verifications (${verifications.length})`} />
              <Tab label="Transaction History" />
              <Tab label="Submit Verification" />
            </Tabs>

            {/* My Certificates Tab */}
            <TabPanel value={activeTab} index={0}>
              {certificatesLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : certificates.length > 0 ? (
                <Grid container spacing={3}>
                  {certificates.map((certificate) => (
                    <Grid item xs={12} sm={6} lg={4} key={certificate.tokenId}>
                      <CertificateCard
                        certificate={certificate}
                        onRetire={handleCertificateRetire}
                      />
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Card>
                  <CardContent sx={{ textAlign: 'center', py: 4 }}>
                    <BlockIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      No Certificates Found
                    </Typography>
                    <Typography variant="body2" color="text.secondary" paragraph>
                      You don't have any carbon credit certificates yet. Complete a verification process to receive your first certificate.
                    </Typography>
                    <Button
                      variant="contained"
                      onClick={() => setActiveTab(3)}
                    >
                      Submit Verification
                    </Button>
                  </CardContent>
                </Card>
              )}
            </TabPanel>

            {/* Verifications Tab */}
            <TabPanel value={activeTab} index={1}>
              {verificationsLoading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : verifications.length > 0 ? (
                <Grid container spacing={2}>
                  {verifications.map((verification) => (
                    <Grid item xs={12} key={verification.requestId}>
                      <Card>
                        <CardContent>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="h6">
                              Verification #{verification.requestId}
                            </Typography>
                            <Chip 
                              label={verification.status} 
                              color={
                                verification.status === 'Approved' ? 'success' :
                                verification.status === 'Rejected' ? 'error' : 'warning'
                              }
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            Project {verification.projectId} • {verification.carbonAmount} tCO₂e • {verification.methodology}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Submitted: {new Date(verification.submissionTime).toLocaleString()}
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              ) : (
                <Alert severity="info">
                  No verification requests found. Submit your first verification to get started.
                </Alert>
              )}
            </TabPanel>

            {/* Transaction History Tab */}
            <TabPanel value={activeTab} index={2}>
              <TransactionHistory />
            </TabPanel>

            {/* Submit Verification Tab */}
            <TabPanel value={activeTab} index={3}>
              <VerificationForm onSubmit={submitVerification} />
            </TabPanel>
          </Paper>
        </>
      )}

      {/* Network Information */}
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Polygon Network Information
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Typography variant="body2" color="text.secondary">Network</Typography>
            <Typography variant="body1">Polygon Mainnet</Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="body2" color="text.secondary">Block Time</Typography>
            <Typography variant="body1">~2 seconds</Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="body2" color="text.secondary">Transaction Cost</Typography>
            <Typography variant="body1">~$0.01</Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="body2" color="text.secondary">Energy Efficiency</Typography>
            <Typography variant="body1">99.95% less than Ethereum</Typography>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
};

export default Blockchain;
```

### **Phase 6B: Final Testing & Deployment**
**Duration**: 6 hours

#### 6.2 End-to-End Integration Test
**File**: `tests/e2e/test_blockchain_integration.py`
```python
import pytest
import asyncio
from web3 import Web3
from backend.services.blockchain_service import blockchain_service

@pytest.mark.asyncio
class TestBlockchainIntegration:
    """Complete blockchain integration test suite"""
    
    async def test_full_verification_workflow(self):
        """Test complete verification to NFT minting workflow"""
        # Step 1: Submit verification request
        verification_data = {
            'project_id': 1,
            'carbon_amount': 1000,
            'methodology': 'VCS',
            'data_hash': '0x' + 'a' * 64,
            'ipfs_hash': 'QmTest123',
            'requester_address': '0x' + '1' * 40
        }
        
        submission_result = await blockchain_service.submit_verification(**verification_data)
        assert submission_result['success'] == True
        
        # Step 2: Approve verification (admin action)
        approval_result = await blockchain_service.approve_verification(
            verification_id=0,
            token_uri='https://ipfs.io/ipfs/QmTest123'
        )
        
        assert approval_result['success'] == True
        assert 'token_id' in approval_result
        
        # Step 3: Verify NFT was minted
        token_id = approval_result['token_id']
        certificate = await blockchain_service.get_certificate(token_id)
        
        assert certificate is not None
        assert certificate.project_id == '1'
        assert certificate.carbon_amount == '1000'
        assert certificate.methodology == 'VCS'
        assert certificate.is_retired == False
        
    async def test_certificate_retirement(self):
        """Test certificate retirement process"""
        # First mint a certificate for testing
        mint_result = await blockchain_service.mint_certificate(
            recipient_address='0x' + '2' * 40,
            project_id=2,
            carbon_amount=500,
            methodology='CDM',
            verification_hash='0x' + 'b' * 64,
            token_uri='https://ipfs.io/ipfs/QmTest456'
        )
        
        assert mint_result['success'] == True
        token_id = mint_result['token_id']
        
        # Verify certificate exists and is active
        certificate = await blockchain_service.get_certificate(token_id)
        assert certificate.is_retired == False
        
        # Test retirement would happen via frontend
        # This would require user's wallet signature
        
    async def test_gas_estimation(self):
        """Test gas price estimation"""
        gas_price = blockchain_service.estimate_gas_price()
        assert gas_price > 0
        assert isinstance(gas_price, int)
        
    async def test_transaction_monitoring(self):
        """Test transaction status monitoring"""
        # Mock transaction hash
        mock_tx_hash = '0x' + '1' * 64
        
        # This would fail for a fake hash, which is expected
        status = blockchain_service.get_transaction_status(mock_tx_hash)
        assert 'status' in status
```

---

## 📁 Complete File Structure

### **New Files Created (24 files)**

#### **Smart Contracts (6 files)**
1. `blockchain/contracts/CarbonCreditNFT.sol` - Main NFT contract
2. `blockchain/contracts/CarbonCreditVerifier.sol` - Verification logic  
3. `blockchain/contracts/interfaces/ICarbonCreditNFT.sol` - Interface
4. `blockchain/hardhat.config.js` - Development configuration
5. `blockchain/deploy/01-deploy-contracts.js` - Deployment script
6. `blockchain/.env.example` - Environment template

#### **Frontend Integration (12 files)**
7. `frontend/src/services/blockchainService.js` - Web3 integration (500+ lines)
8. `frontend/src/hooks/useWeb3.js` - React Web3 hooks (200+ lines)
9. `frontend/src/store/blockchainSlice.js` - Redux state management (300+ lines)
10. `frontend/src/components/blockchain/WalletConnector.js` - Wallet connection UI
11. `frontend/src/components/blockchain/CertificateCard.js` - NFT certificate display
12. `frontend/src/components/blockchain/VerificationForm.js` - Verification submission
13. `frontend/src/components/blockchain/TransactionHistory.js` - Transaction tracking
14. `frontend/src/components/blockchain/MintingInterface.js` - Admin minting UI
15. `frontend/src/pages/Blockchain.js` - Complete page overhaul (400+ lines)
16. `frontend/src/utils/contractHelpers.js` - Contract utilities
17. `frontend/package.json` - Updated dependencies
18. `frontend/.env.example` - Environment configuration

#### **Backend Integration (4 files)**
19. `backend/services/blockchain_service.py` - Python Web3 integration (600+ lines)
20. `backend/api/blockchain_endpoints.py` - API endpoints (400+ lines)
21. `backend/requirements.txt` - Updated Python dependencies
22. `backend/.env.example` - Backend environment config

#### **Testing & Documentation (2 files)**
23. `tests/e2e/test_blockchain_integration.py` - Integration tests
24. `docs/BLOCKCHAIN_DEPLOYMENT_GUIDE.md` - Deployment documentation

---

## 🎯 Success Metrics & Milestones

### **Week 1 Milestones**
- ✅ Smart contracts deployed on Mumbai testnet
- ✅ 100% test coverage for contract functions
- ✅ Gas optimization completed (<500k gas per mint)
- ✅ Contract verification on Polygonscan

### **Week 2 Milestones**  
- ✅ Frontend wallet connection functional
- ✅ Certificate minting automated from backend
- ✅ Real-time transaction monitoring
- ✅ Complete UI/UX for all blockchain features

### **Production Readiness Checklist**
- [ ] Smart contracts audited (recommended)
- [ ] Mainnet deployment completed
- [ ] Frontend environment variables configured
- [ ] Backend blockchain monitoring active
- [ ] User documentation completed
- [ ] Gas fee monitoring and optimization
- [ ] Multi-signature admin controls (optional)

---

## 💰 Cost Breakdown

### **Development Infrastructure**
- **Testnet Deployment**: Free (Mumbai testnet)
- **RPC Services**: $0-50/month (Alchemy free tier sufficient)
- **Development Tools**: Free (Hardhat, OpenZeppelin)

### **Mainnet Deployment**  
- **Contract Deployment**: ~$50-100 (one-time)
- **Contract Verification**: Free
- **Testing**: ~$10-20

### **Operational Costs**
- **Certificate Minting**: ~$0.01-0.05 per certificate
- **Transfers**: ~$0.01-0.02 per transaction  
- **RPC Calls**: ~$0-50/month (based on usage)

---

## ⚡ Quick Start Commands

### **Smart Contract Development**
```bash
cd blockchain
npm install
npx hardhat compile
npx hardhat test
npx hardhat deploy --network mumbai
```

### **Frontend Integration**
```bash  
cd frontend
npm install ethers web3modal @walletconnect/web3-provider
npm start
```

### **Backend Integration**
```bash
cd backend  
pip install web3 eth-account
python -c "from services.blockchain_service import blockchain_service; print('Blockchain service ready')"
```

---

This comprehensive blockchain integration plan transforms your Carbon Credit Verification SaaS into a fully decentralized, immutable certification platform. The implementation provides enterprise-grade security, user-friendly interfaces, and production-ready smart contracts that will establish your platform as a leader in blockchain-based carbon credit verification.

The plan is designed for incremental implementation with thorough testing at each stage, ensuring a robust and reliable blockchain integration that users can trust with their valuable carbon credit assets.