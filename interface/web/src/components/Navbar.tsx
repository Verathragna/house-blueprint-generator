import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  HomeIcon, 
  WrenchScrewdriverIcon as ToolIcon,
  PhotoIcon,
  UserGroupIcon,
  PhoneIcon,
  CurrencyDollarIcon,
  QuestionMarkCircleIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const navItems = [
    { name: 'Home', path: '/', icon: HomeIcon },
    { name: 'Services', path: '/services', icon: ToolIcon },
    { name: 'Portfolio', path: '/portfolio', icon: PhotoIcon },
    { name: 'About', path: '/about', icon: UserGroupIcon },
    { name: 'Contact', path: '/contact', icon: PhoneIcon },
    { name: 'Pricing', path: '/pricing', icon: CurrencyDollarIcon },
    { name: 'FAQ', path: '/faq', icon: QuestionMarkCircleIcon },
  ];

  return (
    <nav className="bg-white shadow-lg fixed w-full z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex-shrink-0 flex items-center">
              <ToolIcon className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-800">BlueprintPro</span>
            </Link>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex md:items-center md:space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.name}
                to={item.path}
                className="text-gray-600 hover:text-blue-600 px-3 py-2 rounded-md text-sm font-medium transition-colors"
              >
                {item.name}
              </Link>
            ))}
            <Link 
              to="/design" 
              className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
            >
              Start Designing
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
            >
              {isOpen ? <XMarkIcon className="h-6 w-6" /> : <Bars3Icon className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navItems.map((item) => (
              <Link
                key={item.name}
                to={item.path}
                className="text-gray-600 hover:text-blue-600 block px-3 py-2 rounded-md text-base font-medium"
                onClick={() => setIsOpen(false)}
              >
                <div className="flex items-center">
                  <item.icon className="h-5 w-5 mr-2" />
                  {item.name}
                </div>
              </Link>
            ))}
            <Link 
              to="/design"
              className="w-full bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors mt-2 block text-center"
              onClick={() => setIsOpen(false)}
            >
              Start Designing
            </Link>
          </div>
        </div>
      )}
    </nav>
  );
}

export default Navbar;