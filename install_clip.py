#!/usr/bin/env python3
"""
Script to diagnose and install CLIP libraries correctly.
This helps resolve naming conflicts and ensures the right libraries are installed.
"""

import subprocess
import sys
import importlib.util

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Output: {result.stdout}")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run command '{cmd}': {e}")
        return False

def check_clip_installation():
    """Check if CLIP libraries are properly installed"""
    print("=== Checking CLIP installations ===")
    
    # Check for OpenAI CLIP
    try:
        import clip
        if hasattr(clip, 'load'):
            print("✓ OpenAI CLIP is properly installed")
            # Test loading a model
            try:
                model, preprocess = clip.load("ViT-B/32", device="cpu")
                print("✓ OpenAI CLIP model loading works")
                openai_clip_ok = True
            except Exception as e:
                print(f"✗ OpenAI CLIP model loading failed: {e}")
                openai_clip_ok = False
        else:
            print("✗ Wrong 'clip' module found - this is not OpenAI CLIP")
            print("  Available attributes:", dir(clip))
            openai_clip_ok = False
    except ImportError:
        print("✗ OpenAI CLIP not found")
        openai_clip_ok = False
    
    # Check for OpenCLIP
    try:
        import open_clip
        print("✓ OpenCLIP is installed")
        # Test listing models
        try:
            models = open_clip.list_models()
            print(f"✓ OpenCLIP has {len(models)} models available")
            openclip_ok = True
        except Exception as e:
            print(f"✗ OpenCLIP model listing failed: {e}")
            openclip_ok = False
    except ImportError:
        print("✗ OpenCLIP not found")
        openclip_ok = False
    
    return openai_clip_ok, openclip_ok

def install_clip_libraries():
    """Install CLIP libraries"""
    print("\n=== Installing CLIP libraries ===")
    
    # First, uninstall any conflicting 'clip' packages
    print("Removing any conflicting 'clip' packages...")
    run_command("pip uninstall -y clip")
    
    # Install OpenAI CLIP
    print("Installing OpenAI CLIP...")
    success1 = run_command("pip install git+https://github.com/openai/CLIP.git")
    
    # Install OpenCLIP
    print("Installing OpenCLIP...")
    success2 = run_command("pip install open_clip_torch")
    
    # Install additional dependencies
    print("Installing additional dependencies...")
    success3 = run_command("pip install ftfy regex")
    
    return success1 and success2 and success3

def main():
    print("CLIP Installation Diagnosis and Fix Tool")
    print("=" * 50)
    
    # Check current installation
    openai_ok, openclip_ok = check_clip_installation()
    
    if openai_ok and openclip_ok:
        print("\n🎉 Both CLIP libraries are working correctly!")
        return
    
    print(f"\n📋 Status: OpenAI CLIP: {'OK' if openai_ok else 'FAILED'}, OpenCLIP: {'OK' if openclip_ok else 'FAILED'}")
    
    response = input("\nWould you like to reinstall CLIP libraries? (y/n): ")
    if response.lower() in ['y', 'yes']:
        if install_clip_libraries():
            print("\n🔄 Installation complete. Checking again...")
            openai_ok, openclip_ok = check_clip_installation()
            if openai_ok or openclip_ok:
                print("\n🎉 At least one CLIP library is now working!")
            else:
                print("\n😞 Installation still has issues. Please check manually.")
        else:
            print("\n😞 Installation failed. Please check your internet connection and try manually.")

if __name__ == "__main__":
    main() 