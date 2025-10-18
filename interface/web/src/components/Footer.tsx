import React from 'react';
import { Link } from 'react-router-dom';
import { 
  WrenchScrewdriverIcon as ToolIcon,
  EnvelopeIcon,
  PhoneIcon,
  MapPinIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-900 text-gray-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Company Info */}
          <div className="space-y-4">
            <div className="flex items-center">
              <ToolIcon className="h-8 w-8 text-blue-400" />
              <span className="ml-2 text-xl font-bold text-white">BlueprintPro</span>
            </div>
            <p className="text-sm">
              Professional residential blueprint design services helping you visualize and plan your perfect home.
            </p>
            <div className="flex space-x-4">
              <a href="https://facebook.com" className="hover:text-blue-400 transition-colors">Facebook</a>
              <a href="https://twitter.com" className="hover:text-blue-400 transition-colors">Twitter</a>
              <a href="https://instagram.com" className="hover:text-blue-400 transition-colors">Instagram</a>
              <a href="https://linkedin.com" className="hover:text-blue-400 transition-colors">LinkedIn</a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/services" className="hover:text-blue-400 transition-colors">Services</Link>
              </li>
              <li>
                <Link to="/portfolio" className="hover:text-blue-400 transition-colors">Portfolio</Link>
              </li>
              <li>
                <Link to="/about" className="hover:text-blue-400 transition-colors">About Us</Link>
              </li>
              <li>
                <Link to="/pricing" className="hover:text-blue-400 transition-colors">Pricing</Link>
              </li>
              <li>
                <Link to="/faq" className="hover:text-blue-400 transition-colors">FAQ</Link>
              </li>
            </ul>
          </div>

          {/* Services */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Our Services</h3>
            <ul className="space-y-2">
              <li>Custom Blueprint Design</li>
              <li>3D Visualization</li>
              <li>Renovation Planning</li>
              <li>Expert Consultation</li>
              <li>Interior Design</li>
            </ul>
          </div>

          {/* Contact Info */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Contact Us</h3>
            <ul className="space-y-4">
              <li className="flex items-center">
                <MapPinIcon className="h-5 w-5 mr-2 text-blue-400" />
                <span>123 Design Street, Suite 100<br />New York, NY 10001</span>
              </li>
              <li className="flex items-center">
                <PhoneIcon className="h-5 w-5 mr-2 text-blue-400" />
                <a href="tel:+1234567890" className="hover:text-blue-400 transition-colors">
                  (123) 456-7890
                </a>
              </li>
              <li className="flex items-center">
                <EnvelopeIcon className="h-5 w-5 mr-2 text-blue-400" />
                <a href="mailto:info@blueprintpro.com" className="hover:text-blue-400 transition-colors">
                  info@blueprintpro.com
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-800 mt-12 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm">
              © {currentYear} BlueprintPro. All rights reserved.
            </p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <Link to="/privacy" className="text-sm hover:text-blue-400 transition-colors">
                Privacy Policy
              </Link>
              <Link to="/terms" className="text-sm hover:text-blue-400 transition-colors">
                Terms of Service
              </Link>
              <Link to="/sitemap" className="text-sm hover:text-blue-400 transition-colors">
                Sitemap
              </Link>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;