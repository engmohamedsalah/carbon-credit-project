"""
Blockchain Service for Carbon Credit NFT Management
Simple implementation using Web3.py and Polygon network
"""
import json
import os
from typing import Optional, Dict, Any
import logging
from web3 import Web3
# POA middleware is no longer needed in newer Web3.py versions
from eth_account import Account
import requests

logger = logging.getLogger(__name__)

class BlockchainService:
    def __init__(self):
        # Use Polygon Mumbai testnet for development
        self.rpc_url = os.getenv("POLYGON_RPC_URL", "https://rpc-mumbai.maticvigil.com")
        self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # POA middleware is no longer needed in newer Web3.py versions for Polygon
        
        # Load contract ABI
        self.contract_abi = self._load_contract_abi()
        
        # Initialize account if private key provided
        if self.private_key:
            self.account = Account.from_key(self.private_key)
        else:
            self.account = None
            
        # Initialize contract if address provided
        if self.contract_address and self.contract_abi:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=json.loads(self.contract_abi)
            )
        else:
            self.contract = None
    
    def _load_contract_abi(self) -> Optional[str]:
        """Load contract ABI from blockchain_config.json"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), "..", "blockchain_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('abi')
        except Exception as e:
            logger.warning(f"Could not load contract ABI: {e}")
        return None
    
    def is_connected(self) -> bool:
        """Check if connected to blockchain network"""
        try:
            return self.w3.is_connected()
        except:
            return False
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get basic network information"""
        try:
            if not self.is_connected():
                return {"error": "Not connected to network"}
            
            chain_id = self.w3.eth.chain_id
            latest_block = self.w3.eth.get_block('latest')
            
            return {
                "connected": True,
                "chain_id": chain_id,
                "network": "Polygon Mumbai" if chain_id == 80001 else "Polygon Mainnet" if chain_id == 137 else f"Chain {chain_id}",
                "latest_block": latest_block.number,
                "gas_price": self.w3.eth.gas_price
            }
        except Exception as e:
            logger.error(f"Error getting network info: {e}")
            return {"error": str(e)}
    
    def mint_carbon_credit_nft(self, recipient_address: str, project_id: int, 
                              carbon_amount: int, project_name: str, 
                              location: str, verification_hash: str) -> Optional[Dict[str, Any]]:
        """Mint a carbon credit NFT"""
        try:
            if not self.contract or not self.account:
                return {"error": "Contract or account not initialized"}
            
            # Create token URI (simplified - in production would be IPFS)
            token_uri = f"https://api.carbonverify.com/metadata/{project_id}"
            
            # Build transaction
            tx_function = self.contract.functions.mintCarbonCredit(
                recipient_address,
                project_id,
                carbon_amount,
                project_name,
                location,
                verification_hash,
                token_uri
            )
            
            # Get gas estimate
            gas_estimate = tx_function.estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = tx_function.build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': int(gas_estimate * 1.2),  # Add 20% buffer
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign transaction
            signed_txn = self.account.sign_transaction(transaction)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            # Get token ID from logs
            token_id = None
            for log in tx_receipt.logs:
                try:
                    decoded_log = self.contract.events.CarbonCreditMinted().process_log(log)
                    token_id = decoded_log['args']['tokenId']
                    break
                except:
                    continue
            
            return {
                "success": True,
                "transaction_hash": tx_hash.hex(),
                "token_id": token_id,
                "gas_used": tx_receipt.gasUsed,
                "block_number": tx_receipt.blockNumber
            }
            
        except Exception as e:
            logger.error(f"Error minting NFT: {e}")
            return {"error": str(e)}
    
    def get_carbon_credit_info(self, token_id: int) -> Optional[Dict[str, Any]]:
        """Get carbon credit information by token ID"""
        try:
            if not self.contract:
                return {"error": "Contract not initialized"}
            
            # Call contract function
            carbon_credit = self.contract.functions.getCarbonCredit(token_id).call()
            
            return {
                "token_id": token_id,
                "project_id": carbon_credit[0],
                "carbon_amount": carbon_credit[1],
                "project_name": carbon_credit[2],
                "location": carbon_credit[3],
                "verification_date": carbon_credit[4],
                "verification_hash": carbon_credit[5],
                "is_retired": carbon_credit[6]
            }
            
        except Exception as e:
            logger.error(f"Error getting carbon credit info: {e}")
            return {"error": str(e)}
    
    def verify_certificate(self, token_id_or_hash: str) -> Dict[str, Any]:
        """Verify a carbon credit certificate by token ID or transaction hash"""
        try:
            # Simple demo mode - return mock data for common test tokens
            demo_tokens = {
                "123": {
                    "token_id": 123,
                    "project_id": 1,
                    "carbon_amount": 1250,
                    "project_name": "Amazon Reforestation Project",
                    "location": "Amazon Basin, Brazil",
                    "verification_date": 1642204800,  # Jan 15, 2022
                    "verification_hash": "0x7d865e959b2466918c9863afca942d0fb89d7c9ac0c99bafc3749504ded97730",
                    "is_retired": False
                },
                "456": {
                    "token_id": 456,
                    "project_id": 2,
                    "carbon_amount": 2500,
                    "project_name": "Solar Farm Development",
                    "location": "California, USA",
                    "verification_date": 1651363200,  # May 1, 2022
                    "verification_hash": "0x9f4e8c1b2d3a5c6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
                    "is_retired": True
                },
                "789": {
                    "token_id": 789,
                    "project_id": 3,
                    "carbon_amount": 500,
                    "project_name": "Mangrove Restoration",
                    "location": "Maldives",
                    "verification_date": 1659312000,  # Aug 1, 2022
                    "verification_hash": "0xa1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
                    "is_retired": False
                }
            }
            
            # If it's a known demo token, return mock data
            if token_id_or_hash in demo_tokens:
                return demo_tokens[token_id_or_hash]
            
            # Try to parse as token ID first
            try:
                token_id = int(token_id_or_hash)
                if self.contract:
                    return self.get_carbon_credit_info(token_id) or {"error": "Token not found"}
                else:
                    return {"error": "Contract not initialized"}
            except ValueError:
                pass
            
            # Try as transaction hash
            if token_id_or_hash.startswith('0x'):
                if self.w3 and self.w3.is_connected():
                    try:
                        tx_receipt = self.w3.eth.get_transaction_receipt(token_id_or_hash)
                        # Look for minting event in receipt
                        for log in tx_receipt.logs:
                            try:
                                if self.contract:
                                    decoded_log = self.contract.events.CarbonCreditMinted().process_log(log)
                                    token_id = decoded_log['args']['tokenId']
                                    return self.get_carbon_credit_info(token_id)
                            except:
                                continue
                        return {"error": "No carbon credit found in transaction"}
                    except:
                        return {"error": "Transaction not found"}
                else:
                    return {"error": "Not connected to blockchain network"}
            
            return {"error": "Invalid token ID or transaction hash"}
            
        except Exception as e:
            logger.error(f"Error verifying certificate: {e}")
            return {"error": str(e)}

# Global instance
blockchain_service = BlockchainService()